# Importing libraries
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

def column_remove(data):
    data.drop(columns=["day", "station_name", "status", "actdep", "schdep", "scharr",
                       "actarr_date", "scharr_date", "distance", "actarr"],
              inplace=True)

    # Changing data-frame column order
    cols_to_order = ['latemin', 'weekday', 'month']
    new_columns = cols_to_order + (data.columns.drop(cols_to_order).tolist())
    return data[new_columns]

def data_preprocess(data, sub_data):
    data = column_remove(data)
    for k, d in enumerate(sub_data):
        sub_data[k] = column_remove(d)

    # Label encoding and min-max scaling
    ms.fit(data.iloc[:, 0].values.reshape(-1, 1))
    for k, d in enumerate(sub_data):
        d.iloc[:, 0] = ms.transform(d.iloc[:, 0].values.reshape(-1, 1))
        sub_data[k] = d
        for i in range(1, 6):
            le.fit(data.iloc[:, i].values)
            d.iloc[:, i] = le.transform(d.iloc[:, i].values)
            sub_data[k] = d

        reframed = series_to_supervised(sub_data[k], 1, 1)
        reframed.drop(reframed.columns[7:], axis=1, inplace=True)
        sub_data[k] = reframed

    return sub_data


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df_temp = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for j in range(n_in, 0, -1):
        cols.append(df_temp.shift(j))
    names += [('var%d(t-%d)' % (j+1, j)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for j in range(0, n_out):
        cols.append(df_temp.shift(-j))
        if j == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, j)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# File manipulation
train_folder = "data/train"
valid_folder = "data/valid"
test_folder = "data/test"

train_files = glob.glob(os.path.join(train_folder, "*"), recursive=True)
valid_files = glob.glob(os.path.join(valid_folder, "*"), recursive=True)
test_files = glob.glob(os.path.join(test_folder, "*"), recursive=True)

df = pd.DataFrame()
df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_valid = pd.DataFrame()

for f in range(len(train_files)):
    df_train_temp = pd.read_csv(train_files[f])
    df_train = pd.concat([df_train, df_train_temp])
    df_test_temp = pd.read_csv(test_files[f])
    df_test = pd.concat([df_test, df_test_temp])
    df_valid_temp = pd.read_csv(valid_files[f])
    df_valid = pd.concat([df_valid, df_valid_temp])
    df = pd.concat([df, df_train_temp, df_test_temp, df_valid_temp], ignore_index=True)

# Data pre-processing
ms = MinMaxScaler()
le = LabelEncoder()
reframed_train, reframed_test, reframed_valid = data_preprocess(df, [df_train, df_test, df_valid])


# split into train and test sets
train = reframed_train.values
test = reframed_test.values
valid = reframed_valid.values
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
valid_X, valid_y = valid[:, :-1], valid[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, valid_X.shape, valid_y.shape)


model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(valid_X, valid_y), verbose=2,
                    shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.legend()
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = ms.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = ms.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y[:1000])
plt.plot(inv_yhat[:1000])
plt.show()