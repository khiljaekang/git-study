from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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

model = XGBClassifier(max_depth = max_depth, learning_rate = learning_rate,
                    n_estimator = n_estimator, n_jobs = n_jobs,
                    colsample_bytree = colsample_bytree,
                    colsample_bylevel = colsample_bylevel)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('점수 :', score)

print(model.feature_importances_)

# print(model.predict(x_test))

# 점수 : 0.9
# [0.16568029 0.07939342 0.37284935 0.382077  ] 
