import pandas_datareader as web
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

def lr(ticker):
    currentDate = datetime.now()
    startDate = datetime.now() - relativedelta(years=1)

    df = web.DataReader(ticker, data_source="yahoo", start=startDate, end=currentDate)
    # Get the Close Price
    df = df[['Close']]

    # A variable for predicting 'n' days out into the future
    forecast_out = 30

    # Create another column (the target or dependent variable) shifted 'n' units up
    # We shift it by the number of days we want to forecast_out
    df['Prediction'] = df[['Close']].shift(-forecast_out)

    with open('lr_pkl' , 'rb') as f:
        lr = pickle.load(f)

    # Set x_forecast equal to the last 1086 rows (-forecast_out) of the original data set from Close column
    x_forecast = np.array(df.drop(columns='Prediction'))[-forecast_out:]

    # linear regression model predictions for the next 'n' days, where n is the forecast_out (1086)
    lr_prediction = lr.predict(x_forecast)
    return lr_prediction

