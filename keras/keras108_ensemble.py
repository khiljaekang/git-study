#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

#2.모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

x2 = Dense(50)(x1)
output1 = Dense(1)(x2)

x3 = Dense(70)(x1)
x3 = Dense(70)(x3)
output2 = Dense(1, activation='sigmoid')(x3)

model = Model(inputs = input1, outputs= [output1, output2])
model.summary()

#3.훈련
model.compile(loss = ['mse', 'binary_crossentropy'], optimizer='adam', metrics=['mse', 'acc'])

model.fit(x_train, [y1_train, y2_train], epochs=100, batch_size=1)

#4,평가, 예측

loss = model.evaluate(x_train,[y1_train,y2_train])
print("loss: ", loss)

x1_pred = np.array([11,12,13,14])

y_pred = model.predict(x1_pred)
print(y_pred)
'''
input_1 (InputLayer)            (None, 1)            0

__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 100)          200         input_1[0][0]  

__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 100)          10100       dense_1[0][0]  

__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 100)
 10100       dense_2[0][0]
____________________________________________________________________________________________
dense_6 (Dense)                 (None, 70)
 7070        dense_3[0][0]
____________________________________________________________________________________________
dense_4 (Dense)                 (None, 50)
 5050        dense_3[0][0]
____________________________________________________________________________________________
dense_7 (Dense)                 (None, 70)
 4970        dense_6[0][0]
____________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)
 51          dense_4[0][0]
____________________________________________________________________________________________
dense_8 (Dense)                 (None, 1)
 71          dense_7[0][0]
============================================================================================
Total params: 37,612
Trainable params: 37,612
Non-trainable params: 0
'''