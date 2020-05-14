#1.데이터
import numpy as np 
x = np.array(range(1,101))
y = np.array(range(101,201))
#웨이트값은 1 바이어스는 100

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

print(x_train)
print(x_val)
print(x_test)

print(y_train)
print(y_val)
print(y_test)




#2.모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#input_dim=1은 x값 1부터 100까지를 한덩어리로 본다
model.add(Dense(110, input_dim= 1 ))
model.add(Dense(80))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))




#3.훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

#4.평가, 예측

loss, mse = model.evaluate(x_test ,y_test, batch_size=1)
print("loss :", loss)
print("mse :", mse)

# y_pred = model.predict(x_pred)
# print("y_predict :", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

