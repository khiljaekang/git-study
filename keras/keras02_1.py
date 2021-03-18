#1. 데이터 생성
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([2,4,6,8,10,12,14,16,18,20])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense 

model = Sequential() #Sequential의 뜻은 순차적, 연속적이라는 뜻. 이것은 MLP의 레이어가 순차적으로 쌓여가는 것을 의미합니다. MLP(Multi Layer Perceptron) -다층신경망
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # Mean Squared Error (손실함수) 정답에 대한 오류, 정답에 가까울수록 작은 값 
model.fit(x, y, epochs=200, batch_size=100)

#4. 평가,예측
loss, acc = model.evaluate(x, y )

print("acc :", acc)
print("loss : ", loss)

 
