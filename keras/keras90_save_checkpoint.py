#####keras67 copy #####

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 

                                  
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist에서 이미 x_train, y_train으로 나눠져 있는 값 가져오기

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train: ' , y_train[0])                           # 5


print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)        : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                                       # (10000,)



# 데이터 전처리 1. 원핫인코딩 : 당연하다              => y 값  
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )            => x 값                                           
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                     


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(100, (2, 2), input_shape  = (28, 28, 1), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(80, (2, 2), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(60, (2, 2), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(40, (2, 2),padding = 'same'))
model.add(Conv2D(20, (2, 2),padding = 'same'))
model.add(Conv2D(10, (2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))                # 다중 분류

# model.save('./model/model_test01.h5')


#3. 훈련                      
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) # metrics=['accuracy']

modelpath = './model/check-{epoch:02d} - {val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose = 1,
                                  save_best_only=True, save_weights_only=False)
  



# EarlyStopping

es = EarlyStopping(monitor = 'loss', patience = 20, mode= 'auto')

hist = model.fit(x_train, y_train, epochs= 16, batch_size= 64, callbacks = [es,checkpoint],
                                   validation_split=0.2, verbose = 1)
# hist값이 epoch순으로 저장된다.

model.save('./model/model_test01.h5')

#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 64)

loss = hist.history['loss']                       # model.fit 에서 나온 값
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc: ', acc)                 # 평가 전 fit의 값              
print('val_acc: ', val_acc)         # 평가 전 fit의 값
print('loss_acc: ', loss_acc)       # 평가 후 나온 결과 값   

# 1번 그림
plt.subplot(2, 1, 1)                              # (2, 1, 1) 2행 1열의 그림 1번째꺼 / subplot : 2장 그림               
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')                     
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')                  
plt.grid()                                        # 격자 생성
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss','val_loss']) 
plt.legend(loc = 'upper right')                   # legend의 위치(location) 설정/ default = 제이 비어 있는 곳
                                                  # 위에 label이 지정되어서 안써도 된다.

# 2번 그림
plt.subplot(2, 1, 2)                              # (2, 1, 2) 2행 1열의 그림 2번째꺼               
plt.plot(hist.history['acc'])                     
plt.plot(hist.history['val_acc'])                  
plt.grid()                                        # 격자 생성
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()     

# loss_acc:  [0.05899345099994898, 0.98580002784729]