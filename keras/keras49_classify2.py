import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import np_utils
# from keras.utils import to_categorical

#1. 데이터
x = np.array(range(1, 11))
y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])




"""One_Hot_Encoding"""
y = np_utils.to_categorical(y)     
y = y[:, 1:]             
# y = to_categorical(y)
""" 
- 다중분류 모델은 반드시 one_hot_encoding사용
- 해당 숫자에 해당되는 자리만 1이고 나머지는 0으로 채운다.
"""

print(y)                    
print(y.shape)                     # (10, 6)


#2. 모델 구성
model = Sequential()
model.add(Dense(10,activation = 'relu', input_dim = 1))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(5,activation = 'relu'))        
model.add(Dense(5,activation = 'softmax'))
""" 다중 분류는 'softmax' 사용
   : 가장 큰 수 빼고는 전부 0으로 나옴
"""




#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])    # acc : 분류 모델 0 or 1
model.fit(x, y, epochs = 100, batch_size =1)
""" loss = 'categorical_crossentropy' : 다중분류에서 사용 """



#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size= 1)
print('loss :', loss)
print('acc :', acc)


x_pred = np.array([1, 2, 3, 4, 5])
y_predict = model.predict(x_pred, batch_size=1)
print(y_predict)

y_predict = np.argmax(y_predict,axis=1) + 1

print(y_predict)
print(y_predict.shape)                               # (3, 6)
"""x하나 집어 넣으면 6개가 나옴 (one_hot_encoding때문)
   0  1  2   3   4   5
"""
#과제 dim을 6에서 5로 변경 
#y_predict를 실수형을 정수형으로 변경 


