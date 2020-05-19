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
    x, y, shuffle = True, random_state=1,
    train_size =0.8 
)



print(x_train)
print(x_test)

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

input1 = Input(shape =(1, ))           
dense1_1 = Dense(7, activation = 'relu')(input1)
dense1_2 = Dense(10, activation = 'relu' )(dense1_1)
dense1_2 = Dense(5, activation = 'relu')(dense1_2)
output1 = Dense(3)(dense1_2)


model = Model(inputs = input1, outputs = output1)

model.summary()


es = EarlyStopping(monitor = 'loss', mode = 'auto', patience = 10)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=500, batch_size=1,
          validation_split=0.25)
 


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
