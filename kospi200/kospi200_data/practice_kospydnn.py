import numpy as np
import pandas as pd

kospi200 = np.load('./data/kospi200.npy', allow_pickle=True)
samsung = np.load('./data/samsung.npy', allow_pickle=True)

print(kospi200)
print(samsung)
print(kospi200.shape)
print(samsung.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(samsung, 5, 1)
print(x[0,:], "\n", y[0])
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(294,25)
x_test = x_test.reshape(127,25)

####데이터 전처리#####
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])


from keras.models import Sequential
from keras.layers import Dense

#모델 구성
model = Sequential()
model.add(Dense(64, input_shape=(25, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
            batch_size=1, epochs=100, callbacks=[early_stopping])

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)

y_pred = model.predict(x_test_scaled)

for i in range(5):
    print('종가 : ', y_test[i],'/ 예측가 :', y_pred[i], y_pred[i])