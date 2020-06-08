# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential, Model
# from keras.layers import Input, Dropout, Conv2D, Flatten, Dense
# from keras.layers import MaxPool2D
# import matplotlib.pyplot as plt

# #1.데이터
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)              #(60000, 28, 28)
# print(x_test.shape)               #(10000, 28 ,28)
# print(y_train.shape)              #(60000, )
# print(y_test.shape)               #(10000, )  

# # x_train = x_train.reshape(x_train.shape[0],28, 28, 1).astype('float32')/255
# # x_test = x_test.reshape(x_train.shape[0],28, 28, 1).astype('float32')/255
# x_train = x_train.reshape(x_train.shape[0],28*28)/255
# x_test = x_test.reshape(x_train.shape[0],28*28)/255



# #0~255까지의 데이터 255로 나누면 minmax의 효과가 있다.

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train.shape) 


# # print(x_train[0].shape)                                 # (28, 28)
# # plt.imshow(x_train[0], 'gray')                          # '2차원'을 집어넣어주면 수치화된 것을 이미지로 볼 수 있도록 해줌    
# # plt.imshow(x_train[0])                                  # 색깔로 나옴
# # plt.show()               

# #2. 모델

# def build_model(drop=0.5, optimizer= 'adam'):
#     inputs = input(shape=(28*28,), name='input')
#     x = Dense(512, activation='relu', name='hidden1')(inputs)
#     x = Dropout(drop)(x)
#     x = Dense(256, activation='relu', name='hidden2')(x)
#     x = Dropout(drop)(x)
#     x = Dense(128, activation='relu', name='hidden3')(x)
#     x = Dropout(drop)(x)
#     outputs = Dense(10, activation='softmax', name='outputs')(x)
#     model = Model(input = inputs, output = outputs)
#     model.compile(optimizer=optimizer, metrics=['acc'],
#                   loss ='categorical_crossentropy')
#     return model

'''
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np

#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                   # (60000, 28, 28)
print(x_test.shape)                                    # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/225
x_test = x_test.reshape(x_test.shape[0], 28*28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (28*28, ), name = 'input')
    x = Dense(51, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(activation = 'softmax', name = 'output')(x)
    model = Model(inputs - inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
           "drop" :dropout}

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model = KerasClassifier(build_fn=build_model, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(modelestimator = model, param_grid = hyperparameters, cv=3 )
search.fit(x_train, y_train)

print(search.best_params_)
'''


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np

#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                   # (60000, 28, 28)
print(x_test.shape)                                    # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/225
x_test = x_test.reshape(x_test.shape[0], 28*28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (28*28, ), name = 'input')
    x = Dense(51, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# parameter
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)               # 0.1~ 0.5 까지 5번 
    return{'batch_size' : batches, 'optimizer': optimizers, 
           'drop': dropout}                                       # dictionary형태

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier # sklearn에서 쓸수 있도로 keras모델 wrapping
model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters()

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
search = GridSearchCV(model, hyperparameters, cv = 3)            # cv = cross_validation
            #사이킷 런에 맞는 형태로 바꿔줌.
# fit
search.fit(x_train, y_train)

print(search.best_params_)   # serch.best_estimator_
