'''
<앙상블> 
: 각자 모델을 훈련시키고 합치는 것 
'''
#1. 데이터
import numpy as np
x1 = np.transpose([range(1, 101), range(311, 411),range(411,511)])  
x2 = np.transpose([range(711, 811), range(711,811), range(511,611)])

y1 = np.transpose([range(101, 201), range(411,511),range(100)])
      

################## 여기서 부터 수정 #################
from sklearn.model_selection import train_test_split    
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(  
    # x, y, random_state=66, shuffle = True,
    x1, x2, y1, shuffle = False,
    train_size =0.8                                     
    )
   
print(x1_train)
print(x2_train)
print(y1_train)

print(x1.shape)


'''

#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

######### 모델 1 #########
input1 = Input(shape =(3, ))           
dense1_1 = Dense(7, activation = 'relu')(input1)
dense1_2 = Dense(10, activation = 'relu' )(dense1_1)
dense1_2 = Dense(5, activation = 'relu')(dense1_2)

   

######### 모델 2 #########
input2 = Input(shape =(3, )) 
dense2_1 = Dense(8, activation = 'relu')(input2) 
dense2_2 = Dense(12, activation = 'relu')(dense2_1)
dense2_2 = Dense(4, activation = 'relu')(dense2_2)
  

######### 모델 병합#########
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1_2, dense2_2])   # list형태로 묶임 

middle1 = Dense(10)(merge1)
middle1 = Dense(6)(middle1)
middle1 = Dense(6)(middle1)
middle1 = Dense(6)(middle1)
middle1 = Dense(4)(middle1)


######### output 모델 구성 ###########

output1 = Dense(30)(middle1)   
output1_2 = Dense(15)(output1)
output1_3 = Dense(3)(output1_2)  




######### 모델 명시 #########
model = Model(inputs = [input1, input2],
              outputs= output1_3) 

model.summary() 


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit([x1_train, x2_train], 
          [y1_train ], epochs =500, batch_size =1,
        # validation_data = (x_val, y_val)
        validation_split= 0.25, verbose=1
)

#4. 평가,예측
loss = model.evaluate([x1_test, x2_test], 
                      [y1_test,  ], batch_size =1)

# print("model.metrics_names : ", model.metrics_names) 

print("loss : ", loss)                               
# print("mse : ", mse)

# y_pred = model.predict(x_pred)  #눈으로 보기 위한 예측값
# print("y_pred : ", y_pred)

y1_predict   = model.predict([x1_test, x2_test])  
print(y1_predict)




# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
print("RMSE1 : ", RMSE1)



# R2 구하기
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)


print("R2_1 : ", r2_1)

'''


