import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#1.데이터
dataset = pd.read_csv('D:/Study/data/csv/winequality-white.csv', index_col = None, header = 0, sep =  ';')

print(dataset["quality"].value_counts)

# np_dataset = dataset.values

print(dataset.shape) # (4898, 12)

x = dataset.iloc[:,:11]
y = dataset.iloc[:, 11]
print(x.shape)            # (4898, 11)
print(y.shape)            # (4898,)


# scaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# train_test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 33, train_size = 0.8)

print(x_train.shape)
print(y_train.shape)

#2. 모델
model = RandomForestClassifier()    



#3. 훈련
model.fit(x_train, y_train)

#4. 평가
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc: ", acc)              

#acc:  0.7071428571428572
