import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

#1. 데이터 

x_train, x_test, y_train, y_test = np.load('./data/project/car_logo2.npy', allow_pickle=True)
print(x_train.shape)   #(4400, 150, 150, 3)
print(x_test.shape)    #(1000, 150, 150, 3)
print(y_train.shape)   #(4400, 10)
print(y_test.shape)    #(4400, 10)

brands =['hyundai logo', 'kia logo','bmw logo', 'chevolet logo', 'honda logo', 'hyundai logo',
             'kia logo', 'lexus logo', 'toyota logo', 'volvo logo']
nb_classes = len(brands)

x_train = x_train.astype(float)/255
x_test = x_test.astype(float)/255

#2. 모델
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(150,150, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), padding = 'same'))
model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Dropout(0.2))          
model.add(Conv2D(32, (3, 3), padding = 'same')) 
model.add(Flatten())
model.add(Dense(10, activation='softmax'))                # 다중 분류

model.summary()

#.3 훈련

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

modelpath = './data/project/checkpoint/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=200, batch_size=10, 
                 validation_split=0.2,callbacks=[es, cp])


#.4 평가
loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print("val_loss :", loss)
print("val_acc :", acc)

## pyplot 시각화
y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']
y_vacc = hist.history['val_acc']
y_acc = hist.history['acc']

x_len1 = np.arange(len(y_loss))
x_len2 = np.arange(len(y_acc))
plt.figure(figsize=(6,6))

## 1 Loss 그래프
plt.subplot(2,1,1)
plt.plot(x_len1, y_vloss, marker='.', c='red', label='val_loss')
plt.plot(x_len1, y_loss, marker='.', c='blue', label='loss')
plt.legend()
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

## 2 Acc 그래프
plt.subplot(2,1,2)
plt.plot(x_len2, y_vacc, marker='.', c='red', label='val_acc')
plt.plot(x_len2, y_acc, marker='.', c='blue', label='acc')
plt.legend()
plt.title('Acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.grid()

plt.subplots_adjust(hspace=0.4)
plt.show()


