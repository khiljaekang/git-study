import numpy as np
import pandas as pd

np_samsung = np.load('./data/exam/samsung.npy')
np_hite = np.load('./data/exam/hite.npy')
# print(np_samsung)
# print(np_hite)
# print(np_samsung.shape)              # (509, 1)
# print(np_hite.shape)                 # (509, 5)

# print(np_samsung[-10:])
# print(np_hite[-10:])

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset)-1:
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number+1, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(np_samsung, 5, 1)
x2, y2 =split_xy5(np_hite, 5, 1)

0 + 5
5 + 1


# print(x2[0,:],"\n", y2[0])

# print(x2.shape)                    #(505, 5, 5)
# print(y2.shape)                    #(505, 1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=1, test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=1, test_size = 0.3)

# print(x1_train.shape)            #(352, 5, 1)
# print(x1_test.shape)             #(152, 5, 1)
# print(x2_train.shape)            #(352, 5, 5)
# print(x2_test.shape)             #(152, 5, 5)


x1_train = np.reshape(x1_train,
    (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test,
    (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train,
    (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))

####데이터 전처리#####

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

print("x1_train_scaled[0, :]",x1_train_scaled[0, :])
x1_train_scaled = x1_train_scaled.reshape(-1,5,1)
x1_test_scaled = x1_test_scaled.reshape(-1,5,1)
x2_train_scaled = x2_train_scaled.reshape(-1,5,5)
x2_test_scaled = x2_test_scaled.reshape(-1,5,5)

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

# Input 1
######### 모델 1 #########
input1 = Input(shape =(5,1))
dense1_1 = LSTM(60, activation='relu', name='dense1_1')(input1)
dense1_2 = Dense(80, activation='relu', name='dense1_2')(dense1_1)
dense1_3 = Dense(100, activation='relu', name='dense1_3')(dense1_2)
dense1_4 = Dense(120, activation = 'relu')(dense1_3)
   

######### 모델 2 #########
input2 = Input(shape =(5,5)) 
dense2_1 = LSTM(60, activation='relu', name='dense2_1')(input2)
dense2_2 = Dense(80, activation='relu', name='dense2_2')(dense2_1)
dense2_3 = Dense(100, activation='relu', name='dense2_3')(dense2_2)
dense2_4 = Dense(120, activation = 'relu')(dense2_3)
  

######### 모델 병합#########
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1_4, dense2_4])   
middle1 = Dense(120)(merge1) 
middle1 = Dense(100)(middle1) 
middle1 = Dense(80)(middle1) 
 

######### output 모델 구성 ###########
output1 = Dense(60)(middle1)   
output1_2 = Dense(40)(output1)
output1_2 = Dense(20)(output1_2)
output1_3 = Dense(1)(output1_2)

model = Model(inputs = [input1, input2], outputs = output1_3)

model.summary()


from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(patience=20)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, verbose=1,
            batch_size=1, epochs=100, callbacks=[early_stopping])

model.save('./model/test0602_kangkhiljae.h5')

loss, mse = model.evaluate([x1_test_scaled,x2_test_scaled], y1_test, batch_size=1)
print('loss: ', loss)
print('mse: ', mse)


y_pred = model.predict([x1_test_scaled, x2_test_scaled])




for i in range(5):
    print('시가 : ', y1_test[i],'/ 예측가 :', y_pred[i])

samsung=x1[-1:]
hite=x2[-1:]
# print("samsung",samsung)
# # print(samsung.shape)
# print("hite",hite)
# # print(hite.shape)

samsung=samsung.reshape(-1,samsung.shape[1]*samsung.shape[2])
hite=hite.reshape(-1,hite.shape[1]*hite.shape[2])

samsung = scaler.transform(samsung)
hite = scaler2.transform(hite)

samsung=samsung.reshape(-1,5,1)
hite=hite.reshape(-1,5,5)

pre=model.predict([samsung,hite])
print(pre)





    
    
    
    
    
    