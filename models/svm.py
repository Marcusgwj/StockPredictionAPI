import pandas_datareader as web
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

currentDate = datetime.now()
startDate = datetime.now() - relativedelta(years=5)
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

# Create and train the Support Vector Machine (Regressor)
# rbf is radio basis function (kernel)
# The free parameters in the model are C and epsilon.
svr_rbf = SVR(kernel='rbf', C=1e8, gamma=0.001)

svr_rbf.fit(x_train, y_train)

svm_confidence = svr_rbf.score(x_test, y_test)

with open('svm_pkl', 'wb') as files:
    pickle.dump(svr_rbf, files)

