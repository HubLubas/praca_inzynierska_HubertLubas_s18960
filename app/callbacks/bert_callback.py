import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

from layouts.bert_layout import get_available_datasets as bert_datasets
from ml_scripts.bert_predictor import bert_prediction


def update_bert_plot(
    bert_data_file: Path
) -> list[str]:
    fig = go.Figure()
    data_df = pd.read_pickle(str(bert_datasets()[bert_data_file]))
    result = bert_prediction(data_df, ['astrazeneca', 'pfizer'])
    print(result)
    result_df = pd.DataFrame(result)
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
