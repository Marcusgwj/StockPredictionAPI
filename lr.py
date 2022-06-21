import pandas_datareader as web

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from datetime import datetime
from dateutil.relativedelta import relativedelta


def lr(ticker):
    currentDate = datetime.now()
    startDate = datetime.now() - relativedelta(years=10)

    df = web.DataReader(ticker, data_source="yahoo", start=startDate, end=currentDate)
    # Get the Close Price
    df = df[['Close']]

    # A variable for predicting 'n' days out into the future
    forecast_out = 7

    # Create another column (the target or dependent variable) shifted 'n' units up
    # We shift it by the number of days we want to forecast_out
    df['Prediction'] = df[['Close']].shift(-forecast_out)

    # Create the independent data set (X)
    # Convert the dataframe to a numpy array
    # We remove/drop the Prediction column to create the independent data set!
    X = np.array(df.drop(columns='Prediction'))

    # Remove the last 'n' rows, where n is forecast_out
    # Just take note, the last 'n' rows will have NaN in their prediction
    X = X[:-forecast_out]

    # Create the dependent data set (y)
    # Convert the dataframe to a numpy array (All of the values including the NaNs)
    y = np.array(df['Prediction'])

    # Get all of the y values except the last 'n' rows
    y = y[:-forecast_out]

    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train the Linear Regression Model
    lr = LinearRegression()

    # Train the model
    lr.fit(x_train, y_train)

    # Testing model: Score returns the coefficient of determination R^2 of the prediction
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)

    # Set x_forecast equal to the last 1086 rows (-forecast_out) of the original data set from Close column
    x_forecast = np.array(df.drop(columns='Prediction'))[-forecast_out:]

    # linear regression model predictions for the next 'n' days, where n is the forecast_out (1086)
    lr_prediction = lr.predict(x_forecast)
    return lr_prediction

