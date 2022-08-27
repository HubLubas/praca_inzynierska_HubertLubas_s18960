import compare  as com
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np 


def check_series(filename_vader, filename_bert, ticker):
    bert_data = pd.read_pickle(filename_bert)
    
    copy_bert = bert_data 
    copy_bert = copy_bert.reset_index()
    
    dates = copy_bert["Date"]
    dates = dates.iloc[0:100] 

    short_profit_total_vader = 0.0
    medium_profit_total_vader = 0.0
    long_profit_total_vader = 0.0
    
    short_profit_total_bert = 0.0
    medium_profit_total_bert = 0.0
    long_profit_total_bert = 0.0
    
    for date in dates:
        (short_profit_json, medium_profit_json, long_profit_json, vader_prediction) = com.predict_profits(filename_vader, filename_bert, date.strftime('%Y-%m-%d'), ticker)
        vader = pd.read_json(vader_prediction, orient='table')
        vader = vader.iloc[0]['Label']
        vader = np.int16(vader).item()
        
        bert_list = json.dumps(bert_prediction)
        bert = int(bert_list[2])
        
        
        print(vader)
        short_profit = pd.read_json(short_profit_json, orient='table')
        short_profit = float(short_profit.iloc[0]['Close'])
        medium_profit = pd.read_json(medium_profit_json, orient='table')
        medium_profit = float(medium_profit.iloc[0]['Close'])
        long_profit = pd.read_json(long_profit_json, orient='table')
        long_profit = float(long_profit.iloc[0]['Close'])
        
        print('#########')
        print(vader, short_profit, medium_profit, long_profit)
        
        if vader == 1: 
            short_profit_total_vader = short_profit_total_vader + short_profit
            medium_profit_total_vader = medium_profit_total_vader + medium_profit
            long_profit_total_vader = long_profit_total_vader + long_profit
            
        if bert == 1: 
            short_profit_total_bert = short_profit_total_bert + short_profit
            medium_profit_total_bert = medium_profit_total_bert + medium_profit
            long_profit_total_bert = long_profit_total_bert + long_profit
        
        
    return (short_profit_total_vader, medium_profit_total_vader, long_profit_total_vader, short_profit_total_bert, medium_profit_total_bert, long_profit_total_bert)
              
