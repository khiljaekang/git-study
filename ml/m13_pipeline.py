import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#1.데이터
iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                   shuffle = True, random_state= 1)


#2.모델
model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = make_pipeline(MinMaxScaler(), SVC())
# pipeline은 scaler 쓰고 어떤 기법을 쓸지 명시, 모델쓰고, 기법 명시
# make pipeline은 (전처리, 모델

print("acc : ", pipe.score(x_test, y_test))

# acc : 0.9666666666666667

#전처리는 cv범위를 제외한 나머지 부분에 대한 val부분까지 훈련해주기 떄문에 acc가 높다. 
