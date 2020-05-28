# cifar10 색상이 들어가 있다.
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

#1. data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train[0])
print('y_train[0] :',y_train[0])

print(x_train.shape)                # (50000, 32, 32, 3)
print(x_test.shape)                 # (10000, 32, 32, 3)
print(y_train.shape)                # (50000, 1)
print(y_test.shape)                 # (10000, 1)

# x : reshape, minmax 

# y : one hot categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)                 # (50000, 10)


#2. model
input1 = Input(shape = (32, 32, 3))

dense1 = Conv2D(60, (3, 3), activation = 'relu', padding = 'same')(input1)
max1= MaxPooling2D(pool_size = 2)(dense1)
drop1 = Dropout(0.5)(max1)

dense2 = Conv2D(80, (3, 3), activation = 'relu', padding = 'same')(max1)
max2= MaxPooling2D(pool_size = 2)(dense2)
drop2 = Dropout(0.5)(max2)


dense3 = Conv2D(100, (3, 3), activation = 'relu', padding = 'same')(drop2)
drop3 = Dropout(0.5)(dense3)

dense4 = Conv2D(40, (3, 3), activation = 'relu', padding = 'same')(dense3)
dense5 = Conv2D(20, (3, 3), activation = 'relu', padding = 'same')(dense4)
drop4 = Dropout(0.5)(dense5)
flat = Flatten()(dense5)
output1 = Dense(10, activation = 'softmax')(flat)

model = Model(inputs = input1, outputs = output1)


#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs= 50, batch_size = 32,
              validation_split =0.3 )


#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss: ', loss)
print('acc: ', acc)



