from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Dense, Flatten

#모델구성

model = Sequential()
model.add(Conv2D(20,(4,4), input_shape = (20,20,1)))
model.add(Conv2D(10,(3,3), padding = 'valid'))
model.add(Conv2D(10,(3,3), padding = 'same'))
model.add(Conv2D(10,(3,3), padding = 'valid'))
model.add(Conv2D(10,(3,3), padding = 'same'))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))

model.summary()
