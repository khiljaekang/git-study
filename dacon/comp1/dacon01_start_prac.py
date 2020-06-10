import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


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



# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x)
# x= scaler.transform(x)
# np_test = scaler.transform(np_test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1, test_size = 0.3)


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# es = EarlyStopping()

# model = Sequential()
# model.add(Dense(100, input_dim= 71, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(140, activation= 'relu'))
# model.add(Dense(160, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(40, activation= 'relu'))
# model.add(Dense(4))

# model.summary()

model = XGBClassifier()       



model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print('acc: ', acc)

# np_test = pd.DataFrame(np_test)
# print(type(np_test))

print(np_test)
y_pred = model.predict(np_test)
print(y_pred)
# y_pred = pd.DataFrame(y_pred)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = x.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, #x 데이터가 카테고리 값인 경우에는 bar 명령과 barh 명령으로 바 차트(bar chart) 시각화를 할 수 있다.
             align='center')
    plt.yticks(np.arange(n_features), x.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)

plot_feature_importances_cancer(model)
plt.show()



# y_pred = pd.DataFrame({
#   'id' : np.array(range(10000, 20000)),
#   'hhb': y_pred[:,0],
#   'hbo2': y_pred[:, 1],
#   'ca': y_pred[:, 2],
#   'na':y_pred[:, 3]
# })
# y_pred = pd.DataFrame(y_pred)

# y_pred.to_csv("./data/dacon/comp1/y_predict2.csv")
