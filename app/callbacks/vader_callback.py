import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

from layouts.vader_layout import get_available_datasets as vader_datasets
from ml_scripts.vader_predictor import predict_vader_v2


def update_vader_plot(
    vader_data_file: Path
) -> list[str]:
    fig = go.Figure()

    data_df = pd.read_pickle(str(vader_datasets()[vader_data_file]))
    result_df = predict_vader_v2(data_df)
    print(result_df)
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
