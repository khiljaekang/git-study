# 시계열에서 시작 시간이 맞지 않을 경우 '0'으로 채운다.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D,Flatten
from sklearn.svm import SVC

#1. data
x = pd.read_csv('./data/dacon/comp3/train_features.csv', index_col =0, header = 0)
y = pd.read_csv('./data/dacon/comp3/train_target.csv', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp3/test_features.csv', index_col = 0, header = 0)



print(x.shape)           #(1050000, 5)
print(y.shape)           #(2800, 4)
print(test.shape)        #(262500, 5)


x = x.drop('Time', axis =1)
test = test.drop('Time', axis =1)

print(x)
print(test)


x = x.values
y = y.values
x_pred = test.values




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
model = Sequential()   
model.add(Conv1D(200, 2, padding= 'same', input_shape= (375, 4)))
model.add(MaxPooling1D())   
model.add(Conv1D(100, 2, padding= 'same',))
model.add(Dense(102))   
model.add(Dense(100))   
model.add(Dense(80))   
model.add(Dense(50))   
model.add(Dense(10))   
model.add(Dense(4))
model.add(Flatten())
model.add(Dense(4))
model.summary()


# EarlyStopping
# es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 50)

#3. compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32, 
                 validation_split= 0.2 )

#4. evaluate
loss_mspe = model.evaluate(x_test, y_test, batch_size= 32)
print('loss_mspe: ', loss_mspe)

y_pred = model.predict(x_pred)

y_pred = pd.DataFrame(y_pred)

y_pred.to_csv("./data/dacon/comp3/y_predict.csv")









