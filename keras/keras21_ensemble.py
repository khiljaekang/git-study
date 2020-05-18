#1. 데이터
import numpy as np
x1 = np.transpose([range(1, 101), range(311, 411), range(100)])  
y1 = np.transpose([range(711, 811), range(711,811), range(100)])

x2 = np.transpose([range(101, 201), range(411,511), range(100,200)])
y2 = np.transpose([range(501, 601), range(711,811), range(100)])


from sklearn.model_selection import train_test_split    
x1_train, x1_test, y1_train, y1_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x1, y1, shuffle = False,
    train_size =0.8                                     
    )
   
x2_train, x2_test, y2_train, y2_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x2, y2, shuffle = False,
    train_size =0.8                                     
    )    

#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 
# model = Sequential()
# model.add(Dense(5, input_dim = 3 ))
# model.add(Dense(4))
# model.add(Dense(1))

input1 = Input(shape =(3, )) 
dense1_1 = Dense(5, activation= 'relu')(input1)
dense1_1 = Dense(5, activation= 'relu')(dense1_1)
dense1_1 = Dense(5, activation= 'relu')(dense1_1)
dense1_1 = Dense(5, activation= 'relu')(dense1_1)
dense1_2 = Dense(4, activation= 'relu')(dense1_1)


input2 = Input(shape =(3, )) 
dense2_1 = Dense(5, activation= 'relu')(input1)
dense2_1 = Dense(5, activation= 'relu')(dense2_1)
dense2_1 = Dense(5, activation= 'relu')(dense2_1)
dense2_1 = Dense(5, activation= 'relu')(dense2_1)
dense2_2 = Dense(4, activation= 'relu')(dense2_1)


from keras.layers.merge import concatenate #단순하게 붙인다 (병합한다)
merge1 = concatenate([dense1_2, dense2_2])#두개 이상이면 리스트 []를 넣어줘

middle1 = Dense(30)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1) 

##### output 모델구성 #####

output1 = Dense(30)(middle1)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(30)(middle1)
output2_2 = Dense(7)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(inputs=[input1, input2],outputs=[output1_3,output2_3])

model.summary()
'''
model = Model(inputs = input1, outputs= output1) # 함수형 모델이라고 정의 : 시퀀스 모델의 경우 ex) model = Sequential()

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x1_train, y1_train, epochs =100, batch_size =1,
        # validation_data = (x_val, y_val)
          validation_split= 0.25, verbose=1 
)

#4. 평가,예측
loss, mse = model.evaluate(x1_test, y1_test, batch_size =1) 
print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y_predict = model.predict(x1_test)  
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y1_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2 : ", r2)
'''