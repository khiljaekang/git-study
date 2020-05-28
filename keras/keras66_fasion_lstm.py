import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.layers import Dense, LSTM

#1. data
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train[0])
print('y_train[0] :',y_train[0])

print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000, )
print(y_test.shape)                 # (10000, )

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)                 # (60000, 10)

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

#2.모델구성

model = Sequential()
model.add(LSTM(800,activation='relu', input_shape = (1, 28*28)))
model.add(Dropout(0.2))

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


model.summary()




#3. 훈련                      # 다중 분류

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
hist = model.fit(x_train, y_train, epochs= 50, batch_size= 500, 
                 validation_split=0.3 )

     

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 500)
print('loss: ', loss)
print('acc: ', acc)

# loss : 0.37575132058858873
# acc : 0.8809000253677368