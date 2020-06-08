#1.데이터
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

#2.모델구성

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10, input_dim =1, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#3.훈련
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4.실행

acc, loss = model.evaluate(x_test, y_test, batch_size=1)
print("acc ", acc)
print("loss ", loss)

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)



