import pandas_datareader as web
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

def svm(ticker):
    currentDate = datetime.now()
    startDate = datetime.now() - relativedelta(years=1)

    df = web.DataReader(ticker, data_source="yahoo", start=startDate, end=currentDate)
    df = df[['Close']]

    forecast_out = 30

    df['Prediction'] = df[['Close']].shift(-forecast_out)

    x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
    
    with open('svm_pkl', 'rb') as f:
        svr_rbf = pickle.load(f)

    svm_prediction = svr_rbf.predict(x_forecast)
    return svm_prediction
