import numpy as np
#1. 데이터 생성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2.모델구성
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(10, input_dim= 1 , activation='relu'))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))




#3.훈련
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)

#4.실행

loss, acc = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)
