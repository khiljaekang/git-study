#1.데이터
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])

#2.모델구성

model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#3.훈련
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.fit(x, y , epochs=100, batch_size=1)

#4.실행

loss, mse = model.evaluate(x, y, batch_size=1)
print("loss :", loss)
print("mse :", mse)

y_pred = model.predict(x_pred)
print("y_pred :", y_pred)

