import numpy as np
x = np.transpose([range(1, 101), range(311,411), range(100)])
y = np.transpose(range(711,811))

print(x.shape)


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split( 
    x, y, shuffle = False,
    train_size =0.8 
)



print(x_train)
print(x_test)

from keras.models import Sequential 
from keras.layers import Dense

model = Sequential()
model.add(Dense(400, input_dim= 3))
model.add(Dense(503))
model.add(Dense(501))
model.add(Dense(205))
model.add(Dense(107))
model.add(Dense(102))
model.add(Dense(800))
model.add(Dense(100))
model.add(Dense(504))
model.add(Dense(300))
model.add(Dense(2000))
model.add(Dense(200))
model.add(Dense(240))
model.add(Dense(500))
model.add(Dense(2020))
model.add(Dense(1000))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=1, 
          validation_split=0.25,verbose=1)
 


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

#R2 0.5 이하
#layers  5개이상
#노드의 개수 10개 이상
#batch_size 8이하
#epochs는 30이상