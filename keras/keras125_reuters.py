from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 10000, test_split = 0.2)

# print(x_train.shape, x_test.shape)   # (8982, ) (2246, )
# print(y_train.shape, y_test.shape)   # (8982, ) (2246, )

print(x_train[0])
print(y_train[0])
print(len(x_train[0]))

# y의 카테고리의 개수 출력
category = np.max(y_train) + 1
print(category)

# y의 유니크한 값 출력
y_b = np.unique(y_train)
# print(y_b)

y_train_pd = pd.DataFrame(y_train)
b = y_train_pd.groupby(0)[0].count()
# print(b)
# print(b.shape)

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 100, padding = 'pre')
# truncating은 만약 150개의 단어가 있는데 맥스렌이 100이면 50개 컷
# padding이 pre면 앞에서 컷, post면 뒤에서
x_test = pad_sequences(x_test, maxlen = 100, padding = 'pre')

print(x_train[0])
# print(len(x_train[0]))
# print(len(x_train[-1]))

# x_train과 x_test shape
# print(x_train.shape) # (8982, 100)
# print(x_test.shape)  # (2246, 100)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Embedding

model = Sequential()
# model.add(Embedding(1000, 3, input_length = 100))
model.add(Embedding(10000, 3, input_length = 100))

model.add(LSTM(5))
model.add(Dense(46, activation = 'softmax'))

model.summary()

# 3. 컴파일

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 10, batch_size = 100, validation_split = 0.2)

acc= model.evaluate(x_test, y_test, batch_size = 100)[1]
print("acc: ", acc)


# 4. 시각화

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker = '.', c = 'red', label = 'TestSet Loss')
plt.plot(y_loss, marker = '.', c = 'blue', label = 'TrainSet Loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()