#딥러닝 모델을 순차적으로 구성하겠다
#순차적 모델 구성에 Dense 레이어를 추가하겠다
#데이터를 넣기 위한 배열을 numpy로 사용하겠다
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#훈련 할 데이터와 평가용 데이터를 구성했다
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

#model.add(Dense(5, input_dim=1, activation='relu'))이 의미하는 것은 1개의 입력으로 5개의 노드를 출력하겠다
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1, activation='relu'))
#모델 구성을 확인 하는 좋은 방법 summary를 사용했다
model.summary()

#머신이 이해할 수 있도록 컴파일 하겠다
#loss: 함수는 어떤 것을 사용할 것인가? mse = 평균제곱법
#optimizer : 최적화 함수는? adam 옵티마이저를 사용하겠다
#metrics : 어떤방식? accuracy 로 판정하겠다
#validation_data는 머신에게 훈련데이터와 평가데이터를 나눠서 학습과 평가를 하기 위함이다
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100, batch_size=100, validation_data= (x_train, y_train))

loss, acc = model.evaluate(x_test, y_test, batch_size=100)

print("loss ", loss) 
print("acc : ",acc)


