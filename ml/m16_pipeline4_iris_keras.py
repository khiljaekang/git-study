#iris를 케라스로 파이프라인 구성
#당연히 RandomizedSearchCV
#keras 98 참조할것 

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
x, y = load_iris(return_X_y = True)
print(x.shape)               #(150, 4)
print(y.shape)               #(150, )

mas = MaxAbsScaler()
pca = PCA(n_components=1)
# 1-1. preprocessing
pca.fit(x)
x = pca.transform(x)


# 1-3. data split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size = 0.2,
#     shuffle = True, random_state = 77)

print(x_train.shape)            # (120, 1)
print(x_test.shape)             # (30, 1)
print(y_train.shape)            # (120,)
print(y_test.shape)             # (30,)

# 1-4. One Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])
print(y_test[0])

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def kang_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (1, ), name = 'input')
    x = Dense(1, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    # return model

# parameter
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    epochs = [20, 40, 60, 80, 100]
    dropout = [0.1, 0.5, 0.3]
    optimizers = ['rmsprop', 'adam', 'adadelta']  
    return{'model__batch_size' : batches, 'model__epochs': epochs, 'model__drop': dropout}
                                                 # dictionary형태

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier # sklearn에서 쓸수 있도로 keras모델 wrapping
model = KerasClassifier(build_fn = kang_model)

hyperparameters = create_hyperparameters()

pipe = Pipeline([('scaler', MinMaxScaler()), ("model", model)])
# pipe = make_pipeline(MinMaxScaler(), RandomizedSearchCV())

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv = 3, )            # cv = cross_validation
            #사이킷 런에 맞는 형태로 바꿔줌.
# fit
search.fit(x_train, y_train)

print(search.best_params_)   # serch.best_estimator_ 

score = search.score(x_test,y_test)
print("score : ", score)









