from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash

from layouts.technical_analyses_layout import technical_analysys_layout
from callbacks.callbacks import register_callbacks
from layouts.navbar_layout import navbar_layout
from layouts.compare_layout import compare_layout
from layouts.vader_layout import vader_layout
from layouts.bert_layout import bert_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar_layout,
        html.Div(id="page-content", children=[
            html.H1(
                id="H1",
                children="Loading...",
                style={"textAlign": "center", "marginTop": 400},
            )
        ]),
    ]
)


# Create the callback to handle mutlipage inputs
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/tech_analasis":
        return technical_analysys_layout
    if pathname == "/bert":
        return bert_layout
    if pathname == "/vader":
        return vader_layout
    if pathname == "/compare":
        return compare_layout
    else:  # if redirected to unknown link
        return "404 Page Error! Please choose a link"


register_callbacks(app)


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port='80', use_reloader=True, debug=True)
