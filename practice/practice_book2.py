from keras.models import Sequential  # keras안에서 .(가져오기) model 중 import(가장 끝에거 가져오기) Sequential 
from keras.layers import Dense,Dropout       # keras안에서 layer중  Dense형을 사용하겠다.
import numpy as np                   # numpy를 가져와 사용하겠다. numpy를 np라고 줄여 쓰겠다.
#1. 데이터 생성
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

#2. 모델 

model = Sequential()
model.add(Dense(10, input_dim= 1, activation='relu'))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

model.summary()

#3. 훈련

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs= 100, batch_size=1)

#4. 실행

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("loss= ", loss)
print("acc= ", acc)

output = model.predict(x_test)
print("결과물 : |n", output)
