import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




#헤더가 하나 빠지니까 데이터는 10000개 

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print("train.shape : ", train.shape)                   #(10000,75) : x_train, x_test

print("test.shape : ", test.shape)                     #(10000,71) : x_predict 
print("submission.shape : ", submission.shape)         #(10000,4)  : y_predict

print(train.isnull().sum())                      #null값에 대한 summary

train = train.interpolate()
test = test.interpolate()

x = train.iloc[:, :71]                           
y = train.iloc[:, -4:]
print(x.shape)                                   # (10000, 71)
print(y.shape)                                   # (10000, 4)

x = x.fillna(method = 'bfill')
test = test.fillna(method = 'bfill')
np_train = train.values
np_test = test.values                          #넘파이로 바꿈
# print(type(test))

print(type(np_test))
#<class 'numpy.ndarray'>


print(np_test)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x= scaler.transform(x)
np_test = scaler.transform(np_test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size= 0.2)


print(x.shape)
print(y.shape)

x_train = x_train.reshape(-1, 71, 1)
x_test = x_test.reshape(-1, 71, 1)


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

#2. model
model = Sequential()   
model.add(Conv1D(100, 2, activation='relu', padding= 'same', input_shape= (71, 1)))
model.add(MaxPooling1D())   
model.add(Conv1D(80, 2, padding= 'same',))
model.add(Dense(40))   
model.add(Flatten())
model.add(Dense(4))
model.summary()



#3.훈련
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1, batch_size=10, callbacks=[es] 
          )

#4.평가, 예측

loss, mae = model.evaluate(x_test ,y_test, batch_size=10)
print("loss :", loss)
print("mae :", mae)

# np_test = pd.DataFrame(np_test)
# print(type(np_test))

print(np_test)
np_test = np_test.reshape(-1,71,1)
y_pred = model.predict(np_test)
print(y_pred)
# y_pred = pd.DataFrame(y_pred)

y_pred = pd.DataFrame({
  'id' : np.array(range(10000, 20000)),
  'hhb': y_pred[:,0],
  'hbo2': y_pred[:, 1],
  'ca': y_pred[:, 2],
  'na':y_pred[:, 3]
})
y_pred = pd.DataFrame(y_pred)

y_pred.to_csv("./data/dacon/comp1/y_predict3.csv")


#scikit-learn 버전 0.22.1


#서브밋 파일을 만든다.
#y_pred.to_csv(경로)



