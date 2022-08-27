from dash import html, dcc
from datetime import date

from ml_scripts import technical_analysis

metrics_map = {
    "SMA": {"func": technical_analysis.sma, "key": "SMA"},
    "ESMA": {"func": technical_analysis.esma, "key": "ESMA"},
    "TRIPLE MACS": {"func": technical_analysis.triple_macs, "key": "Adj Close"},
    "BUY SELL TRIPLE MACS": {"func": technical_analysis.buy_sell_triple_macs},
    "MACD": {"func": technical_analysis.macd, "key": "MACD"},
    "BUY SELL MACD": {"func": technical_analysis.buy_sell_macd},
    "RSI": {"func": technical_analysis.rsi, "key": "RSI"},
    "ROC": {"func": technical_analysis.roc, "key": "ROC"},
    "BUY SELL BOLLINDER BANDS": {"func": technical_analysis.buy_sell_bollinder_bands},
    "SO": {"func": technical_analysis.so, "key": "%D"},
}


technical_analysys_layout = html.Div(
    id="parent",
    children=[
        html.H1(
            id="H1",
            children="Analiza techniczna",
            style={"textAlign": "center", "marginTop": 40, "marginBottom": 40},
        ),
        html.Label(children="Szerokość obliczania SMA/ESMA:"),
        dcc.Input(
            id="wide-text-input",
            placeholder="Wide value for SMA/ESMA",
            type='number',
            value=20
        ),
        dcc.Input(
            id="ticker-text-input",
            placeholder="Ticker",
            type='text',
            value='AZN.L'
        ),
        html.Label(children="Wskaźnik analizy technicznej:"),
        dcc.Dropdown(
            id="metric-type-dropdown",
            options=list(metrics_map.keys()),
            value="SMA",
        ),
        dcc.DatePickerRange(
            id="technical-date-picker-range",
            min_date_allowed=date(1995, 8, 5),
            max_date_allowed=date.today(),
            start_date=date(2018, 5, 1),
            end_date=date(2019, 10, 31),
        ),
        dcc.Graph(id="tech-analyses-plot"),
    ],
)
