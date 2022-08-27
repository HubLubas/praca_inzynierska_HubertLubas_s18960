from dash import html
import dash_bootstrap_components as dbc

navbar_layout = html.Div(
    [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Analiza techniczna", href="/tech_analasis")),
                dbc.NavItem(dbc.NavLink("BERT model", href="/bert")),
                dbc.NavItem(dbc.NavLink("VADER model", href="/vader")),
                dbc.NavItem(dbc.NavLink("Porównanie modeli", href="/compare")),
            ],
            brand="Panel inwestora",
            brand_href="/tech_analasis"
        ),
    ]
)
