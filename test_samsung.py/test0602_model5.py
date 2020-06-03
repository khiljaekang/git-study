##LSTM  2개 구현

import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from keras.models import Sequential, Model


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    #print(type(aaa))
    return np.array(aaa)

size = 6

#1. 데이터
#npy 불러오기
samsung = np.load('./data/samsung.npy', allow_pickle='True') #객체 배열(object)를 저장할 수 있게 해줌
hite = np.load('./data/hite.npy', allow_pickle='True') #object는 str, int와 같은 데이터 타입

print(samsung.shape)        #(509, 1)
print(hite.shape)           #(509, 5)

samsung = samsung.reshape(samsung.shape[0], )  #(509, )

samsung = (split_x(samsung, size))
print(samsung.shape)        #(504, 6)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]


print(x_sam.shape)          #(504,5)
print(y_sam.shape)          #(504, )



# standardscaler            # scaler 는 2차원만 받음. 
scaler = StandardScaler()
scaler.fit(x_sam)
x_sam = scaler.transform(x_sam)
scaler.fit(hite)
x_hit = scaler.transform(hite)


# PCA
pca =PCA(n_components = 3)                      
pca.fit(x_hit)
x_hit = pca.transform(x_hit)
print(x_hit.shape)                # (509, 3)

# split
x_hit = split_x(x_hit, size)
print(x_hit.shape)                # (504, 6, 3)


x_sam = x_sam.reshape(x_sam.shape[0], x_sam.shape[1], 1)



#2. 모델구성 

input1 = Input(shape=(5,1))
x1 = LSTM(10, activation = 'relu')(input1)
x2 = Dense(40)(x1)
x3 = Dense(60)(x2)
x4 = Dense(100)(x3)
x5 = Dropout(0.2)(x4)
x6 = Dense(60)(x5)
x7 = Dense(40)(x6)
x8 = Dense(10)(x7)

input2 = Input(shape=(6,3))
y1 = LSTM(5, activation = 'relu')(input2)
y2 = Dense(40)(y1)
y3 = Dense(60)(y2)
y4 = Dense(100)(y3)
y5 = Dropout(0.2)(y4)
y6 = Dense(60)(y5)
y7 = Dense(40)(y6)
y8 = Dense(10)(y7)

merge = Concatenate()([x7, y7])

output = Dense(1)(merge)

model = Model(inputs= [input1, input2], output=output)

model.summary()

# #3. 컴파일, 훈련 

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x_sam, x_hit], y_sam, epochs=5)    


