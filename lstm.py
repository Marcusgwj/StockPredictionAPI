import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime
from dateutil.relativedelta import relativedelta


def lstm(ticker):
    current_date = datetime.now()
    start_date = datetime.now() - relativedelta(years=2)

    data = web.DataReader(ticker, data_source="yahoo", start=start_date, end=current_date)

    # Create a new dataframe with only the 'Close column'
    close = data.filter(['Close'])

    # Convert the dataframe to a numpy array
    dataset = close.values

    # 80%  of rows will be used for training data
    training_data_len = math.ceil(len(dataset) * 0.8)

    # Scale the data, 0 and 1 inclusive
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the data into x_train and y_train data sets
    x_train = []  # Independent training variables/features
    y_train = []  # Target variables

    # We're using the past 60 days to predict the 61st day
    # Should this be adjustable by user?
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])  # Position 0 - 59
        y_train.append(train_data[i, 0])  # Position 60

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data set (because LSTM model expects 3D or higher, not just 2D)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    # Should the model be adjustable by user?? Both layers and number of neurons?
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # Add a layer with 50 neurons.
    # return_sequences is True because we're going to add another LSTM layer after this layer.
    # input_shape is the number of time steps (60, which is x_train.shape[1]) and number of features (1)

    model.add(LSTM(50, return_sequences=False))
    # Second layer will also have 50 neurons
    # return_sequences is False because we're not going to use any LSTM layers after this layer.

    model.add(Dense(25))
    # A regular densely connected neural network layer with 25 neurons

    model.add(Dense(1))
    # A layer with just 1 neuron

    # Compile the model

    # Optimizer to improve upon the loss function
    # Loss function is to measure how well the model did on training, how far it is from the real data
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model

    # batch size is the total number of training examples present in the single batch. It's 1 because we don't want to further divide it into more batches.
    # epochs is the number of iterations when the entire set has gone forward and backward
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from index 1243 to end (1628?).

    test_data = scaled_data[training_data_len - 60:]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data into a numpy array, so that we can use it in the LSTM model
    x_test = np.array(x_test)

    # Reshape the data from 2D to 3D, because LSTM model needs 3D
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(
        predictions)  # We're unscaling the values, and we want it to be the same as y_test dataset!

    # Evaluate our model
    # Get the root mean squared error (RMSE), how accurate the model is
    # Should we show the rmse for test data to user?

    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

    # small error in your equation for RMSE. You need to take the mean of the squared residuals rather than the square of the mean of the residuals.

    # This is quite a good value, considering our data is at about ten thousands (the stock price)

    # Plot the data
    train = close[:training_data_len]
    valid = close[training_data_len:]
    valid = valid.assign(Predictions=predictions)

    new_quote = web.DataReader("AAPL", data_source="yahoo", start=start_date, end=current_date)

    # Create a new dataframe
    new_data = new_quote.filter(['Close'])

    # Get the last 60 days closing price values and convert the dataframe to an array
    last_60_days = new_data[-60:].values

    # Scale the data to be values between 0 and 1
    # Notice that we are using variable from further up called 'scaler', the MinMaxScaler
    # Not using fit transform because we want to use the MinMaxScaler when we first train and test the data
    last_60_days_scaled = scaler.transform(last_60_days)

    # Create an empty list
    X_test = []

    # Append the past 60 days into the X_test
    X_test.append(last_60_days_scaled)

    # Convert the X_test data set to a numpy array to use it to LSTM model
    X_test = np.array(X_test)

    # Reshape the data into 3D
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Get the predicted scaled price
    pred_price = model.predict(X_test)

    # Undo the scaling
    pred_price = scaler.inverse_transform(pred_price)

    X_test = list(X_test)
    pred_price = list(pred_price)
    X_test.append(scaler.transform(pred_price))

    def predict(x):
        new_data = new_quote.filter(['Close'])
        last_60_days = new_data[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        pred_price_lst = []
        for i in range(x):
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

    # 1086 is for three years. Can ask user to adjust how long they want it to predict for.
    return predict(7)

