from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV   # CV = cross validation

x, y = load_iris(return_X_y = True)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

# parameter
n_estimator = 150         
learning_rate = 0.08       
colsample_bytree = 0.9     
colsample_bylevel = 0.7   

max_depth = 5              
n_jobs = -1                

parameters = [
    {"n_estimator":[100, 200, 300], "learning_late": [0.1, 0.3, 0.001, 0.01],
     "max_depth":[4, 5, 6]},
    {"n_estimator":[90, 100, 110], "learning_late": [0.1, 0.001, 0.01 ],
     "max_depth":[4, 5, 6], "colsample_bytree": [0.6, 0.9, 1]},
    {"n_estimator":[90, 100, 110], "learning_late": [0.1, 0.001, 0.6 ],
     "max_depth":[4, 5, 6], "colsample_bytree": [0.6, 0.9, 1],
     "comsample_bylevel":[0.6, 0.7, 0.9]}
]


model = GridSearchCV(XGBClassifier(), parameters, cv=5, n_jobs= -1)

model.fit(x_train, y_train)

print("===================================================")
print(model.best_estimator_)
print("===================================================")
print(model.best_params_)
print("===================================================")
print(model.best_index_)

score = model.score(x_test, y_test)
print('점수 :', score)



