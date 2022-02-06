import base64
from io import BytesIO
import time

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms
import torchvision.transforms as T

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from PIL import Image
import requests

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform, device='cpu'):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)
    model.to(device)
    # propagate through the model
    outputs = model(img)
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu()
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, ].cpu(), im.size)
    return probas, bboxes_scaled


def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    return scores, boxes


# COCO classes
CLASSES = [
    'N/A', 'человек', 'велосипед', 'автомобиль', 'мотоцикл', 'свмолет', 'автобус',
    'поезд', 'грузовик', 'лодка', 'светофор', 'пожарный кран', 'N/A',
    'знак стоп', 'счетчик на стоянке', 'скамейка', 'птичка', 'кот', 'собака', 'лошадь',
    'овца', 'корова', 'слон', 'медведь', 'зебра', 'жираф', 'N/A', 'рюкзак',
    'зонтик', 'N/A', 'N/A', 'сумочка', 'галстук', 'чемодан', 'фрисби', 'лыжи',
    'сноуборд', 'мяч', 'летающий змей', ',баскетбольный мяч', 'бейсбольная перчатка',
    'скейтборд', 'доска для серфинга', 'теннисная ракетка', 'бутылка', 'N/A', 'бокал для вина',
    'чашка', 'вилка', 'нож', 'ложка', 'чаша', 'банан', 'яблоко', 'бутерброд',
    'апельсин', 'броколли', 'морковь', 'хотдог', 'пица', 'пончик', 'кекс',
    'стул', 'диван', 'растение в горшке', 'кровать', 'N/A', 'обеденный стол', 'N/A',
    'N/A', 'туалет', 'N/A', 'tv', 'ноутбук', 'мышь', 'пульт', 'клавиатура',
    'телефон', 'микроволновка', 'духовой шкаф', 'тостер', 'раковина', 'холодильник', 'N/A',
    'книга', 'часы', 'ваза', 'ножницы', 'плюшевый мишка', 'фен', 'зубная щетка'
]

# Load model
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detr = torch.jit.load('assets/detect_model.pth').eval()#.to(DEVICE)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(500),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Dash component wrappers
def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)


def Column(children=None, width=1, **kwargs):
    nb_map = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}

    return html.Div(children, className=f"{nb_map[width]} columns", **kwargs)


# plotly.py helper functions
def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title=None):
    img_width, img_height = im.size
    fig = go.Figure()
    # This trace is added to help the autoresize logic work.
    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0, 
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height,))

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])
    
    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])
    
    fig.update_layout(title=title, showlegend=showlegend)

    return fig


def add_bbox(fig, x0, y0, x1, y1, 
             showlegend=True, name=None, color=None, 
             opacity=0.5, group=None, text=None):
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        fill="toself",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        name=name,
        hoverlabel_namelength=0,
        text=text,
        legendgroup=group,
        showlegend=showlegend,
    ))


# colors for visualization
COLORS = ['#fe938c','#86e7b8','#f9ebe0','#208aae','#fe4a49', 
          '#291711', '#5f4b66', '#b98b82', '#87f5fb', '#63326e'] * 50

RANDOM_URLS = open('random_urls.txt').read().split('\n')[:-1]
#print("Running on:", DEVICE)

# Start Dash
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for deployments

app.layout = html.Div(className='container', children=[
    Row(html.H2('Проект "Детекция изображений" Akulov_ID334020584')),

    Row(html.P("Поле для URL ссылки на изображение:")),
    Row([
        Column(width=8, children=[
            dcc.Input(id='input-url', style={'width': '100%'}, placeholder='Вставить URL...'),
        ]),
        Column(html.Button("Готовая ссылка", id='button-random', n_clicks=0), width=2),
        Column(html.Button("Запуск модели", id='button-run', n_clicks=0), width=2)
    ]),

    Row(dcc.Graph(id='model-output', style={"height": "70vh"})),

])


@app.callback(
    [Output('button-run', 'n_clicks'),
     Output('input-url', 'value')],
    [Input('button-random', 'n_clicks')],
    [State('button-run', 'n_clicks')])
def randomize(random_n_clicks, run_n_clicks):
    return run_n_clicks+1, RANDOM_URLS[random_n_clicks%len(RANDOM_URLS)]


@app.callback(
    Output('model-output', 'figure'),
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit')],
    [State('input-url', 'value')])
def run_model(n_clicks, n_submit, url):
    apply_nms = False
    try:
        im = Image.open(requests.get(url, stream=True).raw)
    except:
        return go.Figure().update_layout(title='Incorrect URL')

    tstart = time.time()

    scores, boxes = detect(im, detr, transform)
    scores, boxes = filter_boxes(scores, boxes, confidence=0.7, iou=0.5, apply_nms=apply_nms)

    scores = scores.data.numpy()
    boxes = boxes.data.numpy()

    tend = time.time()

    fig = pil_to_fig(im, showlegend=True, title=f'Время работы = {tend-tstart:.2f}s')
    existing_classes = set()

    for i in range(boxes.shape[0]):
        class_id = scores[i].argmax()
        label = CLASSES[class_id]
        confidence = scores[i].max()
        x0, y0, x1, y1 = boxes[i]

        # only display legend when it's not in the existing classes
        showlegend = label not in existing_classes
        text = f"класс={label}<br>вероятность={confidence:.3f}"

        add_bbox(
            fig, x0, y0, x1, y1,
            opacity=0.7, group=label, name=label, color=COLORS[class_id],
            showlegend=showlegend, text=text,
        )

        existing_classes.add(label)

    return fig

if __name__ == '__main__':
     app.run_server(debug=True)