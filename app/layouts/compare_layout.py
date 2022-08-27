from dash import html, dcc
from datetime import date

from layouts.bert_layout import get_available_datasets as bert_datasets
from layouts.vader_layout import get_available_datasets as vader_datasets


compare_layout = html.Div(
    id="parent",
    children=[
        html.H1(
            id="H1",
            children="Porównanie modeli wraz z predykcją zwrotu",
            style={"textAlign": "center", "marginTop": 40, "marginBottom": 40},
        ),
        html.Label(children=" Pliki z danymi: "),
        dcc.Dropdown(
            id="bert-compare-data-dropdown",
            options=list(bert_datasets().keys()),
            value=list(bert_datasets().keys())[0]
        ),
        dcc.Dropdown(
            id="vader-compare-data-dropdown",
            options=list(vader_datasets().keys()),
            value=list(vader_datasets().keys())[0]
        ),
        html.Label(children=" Znacznik: "),
        dcc.Input(
            id="ticker-compare-text-input",
            placeholder="Ticker",
            type='text',
            value='AMZN',
            debounce=True
        ),
         html.Label(children=" Data podjęcia decyzji: "),
        dcc.DatePickerSingle(
            id='compare-date-picker-single',
            min_date_allowed=date(2005, 1, 1),
            max_date_allowed=date.today(),
            initial_visible_month=date(2018, 5, 16),
            date=date(2018, 5, 16)
        ),
        dcc.Graph(id="compare-plot", config={"toImageButtonOptions": {"format": "jpeg"}}),
        html.Div(id='textarea-state-bert-output', style={'whiteSpace': 'pre-line'}),
        html.Div(id='textarea-state-vader-output', style={'whiteSpace': 'pre-line'})
    ],
)
