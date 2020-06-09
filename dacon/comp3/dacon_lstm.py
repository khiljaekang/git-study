# 시계열에서 시작 시간이 맞지 않을 경우 '0'으로 채운다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#1. data
x = pd.read_csv('./data/dacon/comp3/train_features.csv', index_col =0, header = 0)
y = pd.read_csv('./data/dacon/comp3/train_target.csv', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', index_col = 0, header = 0)


x = x.drop('Time', axis =1)
test = test.drop('Time', axis =1)

print(x)
print(test)

x = x.values
y = y.values
x_pred = test.values

print(x.shape)                      # (1050000, 4)

# scaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(-1, 375, 4)
x_pred = x_pred.reshape(-1, 375, 4)

print(x.shape)                      # (2800, 375, 4)
print(x_pred.shape)                 # (700, 375, 4)
print(y.shape)                      # (2800, 4)



# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 10, train_size = 0.2)


#2. model
input1 = Input(shape=(375, 4))
l = LSTM(50, activation = 'relu')(input1)
l1 = Dropout(0.2)(l)
l2 = Dense(100, activation = 'relu')(l1)
l3 = Dropout(0.2)(l2)
l4 = Dense(120, activation = 'relu')(l3)
l5 = Dropout(0.2)(l1)
l6 = Dense(150, activation = 'relu')(l5)
l7 = Dropout(0.2)(l6)
l8 = Dense(110, activation = 'relu')(l7)
l9 = Dropout(0.2)(l8)
l10 = Dense(50, activation = 'relu')(l9)
l11 = Dropout(0.2)(l10)
l12 = Dense(30, activation = 'relu')(l11)
l13 = Dropout(0.2)(l12)
output = Dense(4, activation = 'relu')(l13)

model = Model(inputs = input1, outputs = output)


# EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50)

#3. compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 64, 
                 validation_split= 0.2, callbacks = [es])

#4. evaluate
loss_mspe = model.evaluate(x_test, y_test, batch_size= 64)
print('loss_mspe: ', loss_mspe)

y_pred = model.predict(x_pred)

y_pred = pd.DataFrame(y_pred)

y_pred.to_csv("./data/dacon/comp3/y_predict.csv")







