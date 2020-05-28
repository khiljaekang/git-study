import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

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

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255 
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255 

#2.모델구성
model = Sequential()
model.add(Conv2D(60,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))  
model.add(Conv2D(80, (3,3),activation='relu',))           
model.add(Conv2D(120, (2,2),activation='relu')) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(80, (2,2),activation='relu'))                
model.add(Conv2D(60, (2,2), activation='relu'))                
model.add(Conv2D(40, (2,2),activation='relu')) 
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten()) 
model.add(Dense(10,activation = 'softmax'))  

model.summary()

#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs= 50, batch_size = 32,
              validation_split =0.3 )


#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss: ', loss)
print('acc: ', acc)

##loss:  0.4829740544674918
##acc:  0.8980000019073486
