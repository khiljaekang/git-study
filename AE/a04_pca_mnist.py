import numpy as np


#1. 데이터
from tensorflow.keras.datasets import mnist
mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                              # (60000, 28, 28)
print(x_test.shape)                               # (10000, )
print(y_train.shape)                              # (60000, )
print(y_test.shape)                               # (10000, )


# x_data전처리 : MinMaxScaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# # y_data 전처리 : one_hot_encoding (다중 분류)
# from keras.utils.np_utils import to_categorical
# y_trian = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape)


# reshape : Dense형 모델 사용을 위한 '2차원'
x_train = x_train.reshape(60000, 28*28 ) 
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape)                              # (60000, 784)
print(x_test.shape)                               # (10000, 784)

X = np.append(x_train, x_test, axis=0 )

print(X.shape)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
best_n_component = np.argmax(cumsum >= 0.99) + 1
print(best_n_component)

