import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist   #(keras에서 제공하는 예재파일들)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train: ', y_train[0])

print(x_train.shape)         #(60000, 28, 28)
print(x_test.shape)          #(10000, 28, 28)           #0~9까지의 숫자 70000장
print(y_train.shape)         #(60000, )
print(y_test.shape)          #(10000, )

print(x_train[3].shape)      #(28, 28)
plt.imshow(x_train[5999], 'gray')
# plt.imshow(x_train[0])
plt.show()


