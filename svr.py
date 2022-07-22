import pandas_datareader as web
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
from sklearn.preprocessing import MinMaxScaler


def svr(ticker):
    currentDate = datetime.now()
    startDate = datetime.now() - relativedelta(years=1)

    df = web.DataReader(ticker, data_source="yahoo", start=startDate, end=currentDate)
    df = df[['Close']]

    forecast_out = 30

    df['Prediction'] = df[['Close']].shift(-forecast_out)
    
    x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_forecast = scaler.fit_transform(x_forecast)
    with open('svr_pkl', 'rb') as f:
        svr_rbf = pickle.load(f)
        
    svr_prediction =  scaler.inverse_transform(svr_rbf.predict(x_forecast).reshape(-1, 1)).flatten()
    return svr_prediction

