from numpy import array    #import numpy as np , x= np.array 와 동일하다.
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping


#1. 데이터

x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])   #(4,3)
y = array([4,5,6,7])                           #(4, )      #스칼라가 4개짜리인 것
y2 = array([[4,5,6,7]])                        #(1,4)                    
y3 = array([[4],[5],[6],[7]])                  #(4,1)


print("x.shape : ", x.shape)  
print("y.shape : ", y.shape)   

# x = x.reshape(4,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)       # x.shape[0]= 4, x.shape[1], 1    //4행3열을 1번씩 작업하겠다.
print(x.shape)

#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) #행무시 // LSTM의 와꾸를 맞추기 위함, 노드의개수 10
model.add(Dense(2))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
model.add(Dense(1))

model.summary()

#과제 첫번째 레이어의 파라미터의 개수가 왜 480이 나오는지 설명해라.

# 480 = 10 + 1 + 1 * 4 = 480  /////num_params = [(num_units + input_dim + 1) * num_units] * 4

#3. 실행
es = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 10)
model.compile(optimizer='adam',loss='mse')
model.fit(x, y, epochs=500, batch_size =1,callbacks = [es] ) #x는 3차원 이기떄문에 x_input을 같이 와꾸를 맞춰줘야함

x_input = array([5, 6, 7])                  #(3, )
x_input = x_input.reshape(1,3,1)
print(x_input)

yhat = model.predict(x_input)
print(yhat)













 