from datetime import date
import plotly.graph_objects as go

from layouts.technical_analyses_layout import metrics_map


def update_technical_plot(
    metric_type: str,
    start_date: date,
    end_date: date,
    ticker: str,
    wide: int = 20
) -> list[str]:
    fig = go.Figure()
    if metrics_map[metric_type]["func"].__name__ in ["sma", "esma"]:
        result_df = metrics_map[metric_type]["func"](ticker, start_date, end_date, wide)
    else:
        result_df = metrics_map[metric_type]["func"](ticker, start_date, end_date)
    print((f"Metrics type: {metric_type}\n"
           f"Date from: {start_date}\n"
           f"Date to: {end_date}\n"
           f"Result:\n{result_df}"))

    for column in result_df.columns:
        fig.add_trace(
                go.Scatter(
                    x=result_df.index,
                    y=result_df[column],
                    mode='lines+markers',
                    name=column
                )
            )
    return fig
