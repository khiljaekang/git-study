import numpy as np

from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 

mnist.load_data()                                         # mnist파일 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist에서 이미 x_train, y_train으로 나눠져 있는 값 가져오기

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)        : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                                       # (10000,)


print(x_train[0].shape)                                   # (28, 28)
# plt.imshow(x_train[0], 'gray')                          # '2차원'을 집어넣어주면 수치화된 것을 이미지로 볼 수 있도록 해줌    
# plt.imshow(x_train[0])                                  # 색깔로 나옴
# plt.show()                                              # 그림으로 보여주기


# 데이터 전처리 1. 원핫인코딩 : 당연하다              => y 값  
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데티어 전처리 2. 정규화( MinMaxScalar )    => x 값                                           
x_train = x_train.reshape(60000, 28, 28).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28).astype('float32') /255.   # 뒤에 ' . '을 써도 된다.                                  
#             cnn 사용을 위한 4차원       # 타입 변환       # (x - min) / (max - min) : max =255, min = 0                                      
#                                         : minmax를 하면 소수점이 되기때문에 int형 -> float형으로 타입변환


#2. 모델 구성
# 0 ~ 9까지 씌여진 크기가 (28*28)인 손글씨 60000장을 0 ~ 9로 분류하겠다. ( CNN + 다중 분류)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM



'''
input1 = Input(shape=(28 ,28, 1))
layer1 = Conv2D(filters = 8, kernel_size = (3, 3),
         padding ='same')(input1)
layer2 = MaxPooling2D(pool_size = (3, 3))(layer1)
layer3 = Dropout(rate = 0.2)(layer2)

layer4 = Conv2D(filters = 16, kernel_size = (3, 3),
         padding ='same')(layer3)
layer5 = MaxPooling2D(pool_size = (2, 2))(layer4)
layer6 = Dropout(rate = 0.2)(layer5)

layer7 = Conv2D(filters = 32, kernel_size = (3, 3),
         padding ='same')(layer6)
layer8 = MaxPooling2D(pool_size = (2, 2))(layer7)

layer9 = Conv2D(filters = 64, kernel_size = (3, 3),
         padding ='same')(layer8)

layer10 = Conv2D(filters =128, kernel_size = (3, 3),
         padding ='same')(layer9)
layer11 = Dropout(rate = 0.2)(layer10)

layer12 = Conv2D(filters = 256, kernel_size = (3, 3),
         padding ='same' )(layer11)
layer13 = Conv2D(filters = 10, kernel_size = (3, 3),
         padding ='same' )(layer12)

layer14 = Flatten()(layer13)
output1 = Dense(10, activation= 'softmax')(layer14)

model = Model(inputs = input1, outputs = output1)
model.summary()
'''


model = Sequential()
model.add(LSTM(8, activation='relu', input_shape = (28, 28))) # LSTM설정 : ( 열, 몇개씩 짤라서 작업할 것인가)
model.add(Dense(16))                       # Hidden layer  
model.add(Dense(32))    
model.add(Dense(64))
model.add(Dropout(rate=0.2))  
model.add(Dense(10))   
model.add(Dense(10, activation='softmax'))   


model.summary()




#3. 훈련                      # 다중 분류

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
hist = model.fit(x_train, y_train, epochs= 20, batch_size= 32, 
                 validation_split=0.2, )

     

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 32)
print('loss: ', loss)
print('acc: ', acc)