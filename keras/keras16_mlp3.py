# 1.데이터
import numpy as np
x = np.transpose(range(1, 101))
y = np.transpose([range(101, 201), range(711,811), range(100)])

print(x.shape)

# a = np.transpose(x)
# b = np.transpose(y)
# print(a)
# print(a.shape)
# print(b)
# print(b.shape)


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split( 
    x, y, shuffle = False,
    train_size =0.8 
)



print(x_train)
print(x_test)
# 2. 모델구성
from keras.models import Sequential 
from keras.layers import Dense

model = Sequential()
model.add(Dense(110, input_dim= 1))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10)) 
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)
 

# 4. 평가 
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
