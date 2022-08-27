import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import json
import numpy as np 

from layouts.vader_layout import get_available_datasets as vader_datasets
from layouts.bert_layout import get_available_datasets as bert_datasets
from ml_scripts.compare import predict_profits


def update_compare_plot(
    bert_data_file: Path,
    vader_data_file: Path,
    date_str: str,
    ticker: str
) -> list[str]:
    fig = go.Figure()
    result_df, bert_value, vader_value = predict_profits(
        str(vader_datasets()[vader_data_file]),
        str(bert_datasets()[bert_data_file]),
        date_str, ticker
    )

    print(result_df)
    fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=result_df.values,
                mode='lines+markers'
            )
        )
    return fig, f'Predykcja Bert:  {bert[1]}', f'Predykcja Vader:  {vader}'
