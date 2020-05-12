import numpy as np

#데이터생성 
#데이터를 넣기 위한 배열을 numpy로 사용했다
x = np.array([5,10,15])
y = np.array([10,20,30])

from keras.models import Sequential
from keras.layers import Dense

#딥러닝 모델을 순차적으로 구성하겠다
#순차적 모델 구성에 Dense 레이어를 추가하겠다
model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))

#머신이 이해할 수 있도록 컴파일 하겠다
#loss: 함수는 어떤 것을 사용할 것인가? mse = 평균제곱법
#optimizer : 최적화 함수는? adam 옵티마이저를 사용하겠다
#metrics : 어떤방식? accuracy 로 판정하겠다
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#케라스의 모델 실행은 fit이다
#epoch값은 몇번을 훈련시킬지, batch size 는 몇개씩 끊어서 작업 할 것인가를 의미한다
#evaluate는 최종결과에 대한 평가이다
model.fit(x, y, epochs=500, batch_size=1)
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss: ", loss)
print("acc :", acc)



