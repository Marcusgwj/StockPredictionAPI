import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from dateutil.relativedelta import relativedelta
from keras.models import load_model



def lstm(ticker):
    current_date = datetime.now()
    start_date = datetime.now() - relativedelta(years=3)
    scaler = MinMaxScaler(feature_range=(0, 1))

    model = load_model('lstm_model.h5')


    def predict(days):
        new_quote = web.DataReader(ticker, data_source="yahoo", start=start_date, end=current_date)
        new_data = new_quote.filter(['Close'])
        last_60_days = new_data[-60:].values
        last_60_days_scaled = scaler.fit_transform(last_60_days)
        pred_price_lst = []
        for i in range(days):
            X_test = []
            X_test.append(last_60_days_scaled)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            pred_price = model.predict(X_test)
            pred_price = scaler.inverse_transform(pred_price)
            pred_price = list(pred_price)
            pred_price_number = pred_price[0]
            pred_price_lst.extend(pred_price_number)

            last_60_days = list(last_60_days)
            last_60_days.extend(pred_price)
            last_60_days.pop(0)
            last_60_days_scaled = scaler.transform(last_60_days)
        return np.array(pred_price_lst)

    return predict(30)