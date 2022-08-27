from pathlib import Path
from dash import html, dcc


def get_available_datasets():
    files = Path('models')
    bert_files = list(files.glob('bert_data*.pkl'))
    return {bf.name: bf for bf in bert_files}


bert_layout = html.Div(
    id="parent",
    children=[
        html.H1(
            id="H1",
            children="Analiza sentymentu BERT",
            style={"textAlign": "center", "marginTop": 40, "marginBottom": 40},
        ),
        html.Label(children=" Pliki z danymi: "),
        dcc.Dropdown(
            id="bert-data-dropdown",
            options=list(get_available_datasets().keys()),
            value=list(get_available_datasets().keys())[0]
        ),
        dcc.Graph(id="bert-plot"),
    ],
)
