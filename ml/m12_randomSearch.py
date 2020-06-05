# RandomSearch 적용
#mnist적용

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score   # Kfold : 교차 검증
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV   # CV = cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer


# gridSearch
# 내가 정해놓은 조건들을 충족하는 것을 전부다 가져온다. 


#1. data
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)              # (569, 30)
print(y.shape)              # (569,)

print(x)   #(569, 30)
print(y)   #(569,)



# 1-2. split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)
print(x_train.shape)                # (455, 25)
print(x_test.shape)                 # (114, 25)
print(y_train.shape)                # (455,)
print(y_test.shape)                 # (114,)



parameters =  {"n_estimators" : [100, 200],     
    "max_depth": [6, 8, 10, 20],         
    "min_samples_leaf":[3, 5, 7, 10],
    "min_samples_split": [2,3,5]} 
                                      
       # 가중치 인수                                  # = running rate

kfold = KFold(n_splits = 5, shuffle = True)                                                  # train, validation

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv =  kfold)  # SVC()모델을 가지고 parameters를 조정하고, kfold만큼 
     #  진짜 모델,     그 모델의 파라미터 , cross_validtion 수


model.fit(x_train, y_train)

print('최적의 매개변수 : ', model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred) )
