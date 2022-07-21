import pandas_datareader as web
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

def lr(ticker):
    currentDate = datetime.now()
    startDate = datetime.now() - relativedelta(years=1)

    df = web.DataReader(ticker, data_source="yahoo", start=startDate, end=currentDate)
    df = df[['Close']]

    forecast_out = 30

    df['Prediction'] = df[['Close']].shift(-forecast_out)

    with open('lr_pkl', 'rb') as f:
        lr = pickle.load(f)

    x_forecast = np.array(df.drop(columns='Prediction'))[-forecast_out:]

    lr_prediction = lr.predict(x_forecast)
    return lr_prediction

