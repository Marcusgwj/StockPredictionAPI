import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime
from dateutil.relativedelta import relativedelta


current_date = datetime.now()
start_date = datetime.now() - relativedelta(years=10)
ticker = "AAPL"

data = web.DataReader(ticker, data_source="yahoo", start=start_date, end=current_date)

close = data.filter(['Close'])

dataset = close.values

training_data_len = math.ceil(len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]

x_train = []  
y_train = []  

# We're using the past 60 days to predict the 61st day
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])  
    y_train.append(train_data[i, 0])  

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data set (because LSTM model expects 3D or higher, not just 2D)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

# Add a layer with 50 neurons.
# return_sequences is True because we're going to add another LSTM layer after this layer.
# input_shape is the number of time steps (60, which is x_train.shape[1]) and number of features (1)
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# Second layer will also have 50 neurons
# return_sequences is False because we're not going to use any LSTM layers after this layer.
model.add(LSTM(50, return_sequences=False))

# A regular densely connected neural network layer with 25 neurons
model.add(Dense(25))

# A layer with just 1 neuron
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# batch size is the total number of training examples present in the single batch. 
# epochs is the number of iterations when the entire set has gone forward and backward
model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_len - 60:]

x_test = []
y_test = dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # unscaling the values

rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

model.save("lstm_model.h5")