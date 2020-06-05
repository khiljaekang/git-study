import numpy as np

from keras.models import Sequential
from keras.layers import Dense

x = np.array([1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010])
y = np.array([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010])

model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(400))
model.add(Dense(600))
model.add(Dense(800))
model.add(Dense(1000))
model.add(Dense(800))
model.add(Dense(600))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=1)

loss,acc =model.evaluate(x, y, batch_size=1)

print("acc:", acc)
print("loss:", loss)

#데이터가 많지 않기 때문에 accuracy가 0일수도 있는건가?
