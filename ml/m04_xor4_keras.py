import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC        
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


#1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]          # [0, 0] 이 들어가서 [0]
y_data = [0, 1, 1, 0]

x_data =np.array(x_data)
y_data =np.array(y_data)

print(x_data.shape)         #(4,2)
print(y_data.shape)         #(4, )

#2. 모델

model=Sequential()
model.add(Dense(10, input_shape =(2, ), activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

#3. 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data, y_data, epochs= 100, batch_size=1)

#4. 평가,예측
loss, acc = model.evaluate(x_data, y_data, batch_size = 1) 

print("acc : ", acc)





