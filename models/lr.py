import pandas_datareader as web
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

currentDate = datetime.now()
startDate = datetime.now() - relativedelta(years=10)
ticker = "AAPL"

df = web.DataReader(ticker, data_source="yahoo", start=startDate, end=currentDate)
df = df[['Close']]

forecast_out = 30

df['Prediction'] = df[['Close']].shift(-forecast_out)

X = np.array(df.drop(columns='Prediction'))

X = X[:-forecast_out]

y = np.array(df['Prediction'])

y = y[:-forecast_out]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()

lr.fit(x_train, y_train)

lr_confidence = lr.score(x_test, y_test)

with open('lr_pkl', 'wb') as files:
    pickle.dump(lr, files)

