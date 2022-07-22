import pandas_datareader as web
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
from sklearn.preprocessing import MinMaxScaler

currentDate = datetime.now()
startDate = datetime.now() - relativedelta(years=2)
ticker = "MSFT"
    
df = web.DataReader(ticker, data_source="yahoo", start=startDate, end=currentDate)
df = df[['Close']]

forecast_out = 30

df['Prediction'] = df[['Close']].shift(-forecast_out)

X = np.array(df.drop(columns='Prediction'))

X = X[:-forecast_out]

y = np.array(df['Prediction'])

y = y[:-forecast_out]

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)

svr_rbf = SVR(kernel='linear', C=1e7, gamma='auto')

svr_rbf.fit(x_train, y_train)

svr_confidence = svr_rbf.score(x_test, y_test)

with open('svr_pkl', 'wb') as files:
    pickle.dump(svr_rbf, files)

