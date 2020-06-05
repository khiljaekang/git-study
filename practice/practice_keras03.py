import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,4,6,8,10,12,14,16,18,20])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)

loss, acc = model.evaluate(x, y, batch_size=1)

print("loss:", loss)
print("acc: ", acc)

#20번 실행했을 때 19번 acc값이 1.0 이었다



