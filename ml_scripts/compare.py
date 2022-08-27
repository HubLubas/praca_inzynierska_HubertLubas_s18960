from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from ml_scripts import vader_predictor as vp
from ml_scripts import bert_predictor as bp


def data_handler(date, period):
    new_date = (date + timedelta(days=period)).strftime("%Y-%m-%d")
    return new_date


# TO DO ENDPOINT
def predict_profits(filename_vader, filename_bert, decision_date, ticker):
    date_desicion = datetime.strptime(decision_date, "%Y-%m-%d").date()

    data_end = yf.download(
        ticker, start=decision_date
    )
    data_end = data_end["Close"]

    is_trading_day = 0
    i = 1

    while is_trading_day == 0:
        long_date = data_handler(date_desicion, 360 + i)
        print(f'Looking for a long term data {long_date}')
        value = yf.download(ticker, start=long_date, end=(datetime.strptime(long_date, "%Y-%m-%d").date() + timedelta(days=1)).strftime("%Y-%m-%d"))["Close"]
        print(len(value))
        if len(value) == 0:
            i = i + 1
        else:
            is_trading_day = 1

    is_trading_day2 = 0
    j = 1
    while is_trading_day2 == 0:
        medium_date = data_handler(date_desicion, 180 + j)
        print(f'Looking for a medium term data {medium_date}')
        mvalue = yf.download(ticker, start=medium_date, end=(datetime.strptime(medium_date, "%Y-%m-%d").date() + timedelta(days=1)).strftime("%Y-%m-%d"))["Close"]
        print(len(mvalue))
        if len(mvalue) == 0:
            j = j + 1
        else:
            is_trading_day2 = 1

    is_trading_day3 = 0
    k = 1
    while is_trading_day3 == 0:
        short_date = data_handler(date_desicion, 30 + k)
        print(f'Looking for a short term data {short_date}')
        svalue = yf.download(ticker, start=short_date, end=(datetime.strptime(short_date, "%Y-%m-%d").date() + timedelta(days=1)).strftime("%Y-%m-%d"))["Close"]
        print(len(svalue))
        if len(svalue) == 0:
            k = k + 1
        else:
            is_trading_day3 = 1

    long_profit = value - data_end.loc[date_desicion.strftime("%Y-%m-%d")]
    medium_profit = mvalue - data_end.loc[date_desicion.strftime("%Y-%m-%d")]
    short_profit = svalue - data_end.loc[date_desicion.strftime("%Y-%m-%d")]

    result_df = pd.concat([short_profit, medium_profit, long_profit])
    print(f'Profits dataframe:\n{result_df}')

    bert_data = pd.read_pickle(filename_bert)
    vader_data = pd.read_pickle(filename_vader)

    bert_data = bert_data.loc[[decision_date]]
    vader_data = vader_data.loc[[decision_date]]

    bert_prediction = bp.bert_prediction(bert_data, ["astrazeneca", "pfizer"])
    print(f'Bert result: {bert_prediction}')
    vader_prediction = vp.predict_vader_v2(vader_data, "models/model.pkl")
    print(f'Vader result: {vader_prediction}')

    return result_df, bert_prediction, vader_prediction

