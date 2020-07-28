import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import DenseNet169, DenseNet201, NASNetMobile, MobileNet,Xception,ResNet101V2,InceptionV3
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
import efficientnet.tfkeras as efn

start = time.time()

# load data
x = np.load('D:/teamproject/data/face_image_total.npy')
y = np.load('D:/teamproject/data/face_label_total.npy')

print(x.shape) # (7160, 112, 112, 3)
print(y.shape) # (7160, 11)
print('data_load 걸린 시간 :', time.time() - start)
print('======== data load ========')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state = 66)

# model
takemodel = efn.EfficientNetB6(include_top=False, input_shape = (128, 128, 3),)
# takemodel.trainable=False
model = Sequential()
model.add(takemodel)
model.add(Flatten())
# model.add(Dense(300,activation="relu"))
model.add(Dense(12, activation = 'softmax'))
optimizer = SGD(lr=0.001)

model.summary()

cp = ModelCheckpoint('D:/teamproject/checkpoint/EfficientNetB6_64.hdf5', monitor = 'val_loss',
                    save_best_only = True, save_weights_only = False)
es = EarlyStopping(monitor= 'val_loss', patience = 25, verbose =1)

#3. compile, fit
model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])                             
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 64, verbose = 1, 
                 validation_split =0.2 , shuffle = True, callbacks = [es, cp])

# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)
#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)
print('loss_acc: ' ,loss_acc)

end = time.time()
print('총 걸린 시간 :', end-start)

import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = '^', c = 'magenta', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '^', c = 'cyan', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = '^', c = 'magenta', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '^', c = 'cyan', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.show()