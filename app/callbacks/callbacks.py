from dash import Input, Output
from datetime import date, datetime

from callbacks.technical_analysys_callback import update_technical_plot
from callbacks.bert_callback import update_bert_plot
from callbacks.vader_callback import update_vader_plot
from callbacks.compare_callback import update_compare_plot


def register_callbacks(app):
    @app.callback(
        Output("tech-analyses-plot", "figure"),
        Input("metric-type-dropdown", "value"),
        Input("technical-date-picker-range", "start_date"),
        Input("technical-date-picker-range", "end_date"),
        Input("ticker-text-input", "value"),
        Input("wide-text-input", "value"),
    )
    def update_tech_analysis_graph(
        metrics_type: str, start_date: str, end_date: str, ticker: str, wide: str
    ) -> list[str]:
        if start_date is not None:
            start_date_object = date.fromisoformat(start_date)
            start_date_string = start_date_object.strftime("%Y-%m-%d")
        if end_date is not None:
            end_date_object = date.fromisoformat(end_date)
            end_date_string = end_date_object.strftime("%Y-%m-%d")
        return update_technical_plot(
            metrics_type, start_date_string, end_date_string, ticker, wide
        )

    @app.callback(Output("bert-plot", "figure"),
                  Input("bert-data-dropdown", "value"))
    def update_bert_graph(bert_data_file: str) -> list[str]:
        return update_bert_plot(bert_data_file)

    @app.callback(Output("vader-plot", "figure"),
                  Input("vader-data-dropdown", "value"))
    def update_vader_graph(vader_data_file: str) -> list[str]:
        return update_vader_plot(vader_data_file)

    @app.callback(Output("compare-plot", "figure"),
                  Output('textarea-state-bert-output', 'children'),
                  Output('textarea-state-vader-output', 'children'),
                  Input("bert-compare-data-dropdown", "value"),
                  Input("vader-compare-data-dropdown", "value"),
                  Input("compare-date-picker-single", "date"),
                  Input("ticker-compare-text-input", "value"))
    def update_compare_graph(bert_data_file: str,
                             vader_data_file: str,
                             compare_date: datetime.date,
                             ticker: str) -> list[str]:
        if compare_date is not None:
            compare_date_object = date.fromisoformat(compare_date)
            compare_date_string = compare_date_object.strftime("%Y-%m-%d")
        return update_compare_plot(bert_data_file, vader_data_file,
                                   compare_date_string, ticker)
