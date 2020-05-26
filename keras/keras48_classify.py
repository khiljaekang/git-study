"""train_test_predict분리하지 말고"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array(range(1, 11))
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) 

print(x.shape) # (10,) 스칼라가 10개 벡터가 1개 = input_dim=1 
print(y.shape) # (10,)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 1, activation='relu')) 
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #output에 activation='sigmoid'


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x, y, epochs = 100, batch_size = 1, )

#분류모델에서는 metrics를 accuracy를 쓴다. 
#2진분류에서 loss 값은 'binary_crossentropy'


#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size= 1)
print('loss :', loss)
print('acc :', acc)

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)
print('y_pred :', y_pred)

# y_predict = model.predict(x)
# print(y_predict)

'''
#################과제##################
 0과 1이 나오게 만들어라
 1.잘만들어온 걸 퍼오던지
 2.sigmoid를 함수로 불러오던지
 3.기타등등
'''

