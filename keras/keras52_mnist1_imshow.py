import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist   #(keras에서 제공하는 예재파일들)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train: ', y_train[0])

print(x_train.shape)
print(x_test.shape)                     #0~9까지의 숫자 70000장
print(y_train.shape)
print(y_test.shape)

print(x_train[0].shape)
plt.imshow(x_train[20], 'gray')
# plt.imshow(x_train[0])
plt.show()


