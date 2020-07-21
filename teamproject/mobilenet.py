import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import DenseNet169, DenseNet201, NASNetMobile, MobileNet,Xception
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

start = time.time()

# load data
x = np.load('D:/teamproject/data/dog_img224.npy')
y = np.load('D:/teamproject/data/dog_label224.npy')

print(x.shape) # (7160, 112, 112, 3)
print(y.shape) # (7160, 11)
print('data_load 걸린 시간 :', time.time() - start)
print('======== data load ========')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state = 66)

# model
takemodel = DenseNet201(include_top=False, input_shape = (224, 224, 3))

model = Sequential()
model.add(takemodel)
model.add(Flatten())
model.add(Dense(12, activation = 'softmax'))

model.summary()

cp = ModelCheckpoint('D:/teamproject/checkpoint/best_Xception_2.hdf5', monitor = 'val_loss',
                    save_best_only = True, save_weights_only = False)
es = EarlyStopping(monitor= 'val_loss', patience = 25, verbose =1)

#3. compile, fit
model.compile(optimizer = Adam(1e-4), loss = 'categorical_crossentropy', metrics = ['acc'])                             
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 2, verbose = 1, 
                 validation_split =0.2 , shuffle = True, callbacks = [es, cp])


#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 2)
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
plt.ylabel('loss')
plt.legend()

plt.show()
