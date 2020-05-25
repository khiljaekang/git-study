from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping

# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])                      # (13, )   벡터
x_predict = array([50, 60, 70])                                    # (3, )

print('x.shape : ',x.shape)               # (13, 3)
print('y.shape : ',y.shape)               # (13, ) != (13, 1)
                                          #  벡터      행렬

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)  
print(x.shape)                            
'''
                행            열        몇개씩 자르는 지
x.shape = ( batch_size , time_steps , feature )
input_shape = (time_steps, feature )
input_length = timesteps
input_dim = feature
                 x      | y
            ---------------- 
batch_size   1   2   3  | 4     : x의 한 행에 들어간 값을 몇개씩 자르느냐 = feature
             2   3   4  | 5       ex) feature 1 : [1], [2], [3]
             3   4   5  | 6       ex) feature 3 : [1, 2, 3]
             4   5   6  | 7 
              time_step
'''

#2. 모델구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape = (3, 1)))
#input1 = Input(shape=(3,))


input1 = Input(shape=(3, 1))
dense1 = LSTM(10)(input1)
dense2 = Dense(100)(dense1)
dense2 = Dense(100)(dense2)
dense2 = Dense(100)(dense2)
dense2 = Dense(100)(dense2)
dense2 = Dense(100)(dense2)
dense2 = Dense(100)(dense2)
dense2 = Dense(100)(dense2)
output1 = Dense(1)(dense2)

model = Model(inputs = input1, outputs = output1)


model.summary()
'''
LSTM_parameter 계산
num_params = 4 * ( num_units   +   input_dim   +   1 )  *  num_units
                (output node값)  (잘라준 data)   (bias)  (output node값)
           = 4 * (    5      +       1       +   1 )  *     5          = 140     
                    역전파 : 나온 '출력' 값이 다시 '입력'으로 들어감(자귀회귀)
'''


# EarlyStopping
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')

#3. 실행
es = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 10)
model.compile(optimizer='adam', loss = 'mse')
model.fit(x, y, epochs =1000, batch_size = 32,callbacks = [es]) #callbacks = [es] )                

#4. 예측

x_predict = x_predict.reshape(1, 3, 1)       # x값 (4, 3, 1)와 동일한 shape로 만들어 주기 위함
                                         # (1, 3, 1) : 확인 1 * 3 * 1 = 3
# x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
