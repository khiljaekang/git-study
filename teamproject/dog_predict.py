import os
import cv2
import glob
from tensorflow.keras.models import load_model
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
# import sklearn as sk
import h5py
import numpy as np

img = cv2.imread('D:/test/park.jpg')
img = cv2.resize(img, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)
img = img.reshape(-1, 256, 256, 3 ) /255.0
x = img

print(x.shape)

# x_train, x_test, y_train, y_test = sk.preprocessing.train_test_split(
#     x, y, test_size = 0.2)

model = load_model('D:/checkpoint/efficientnet_true2.hdf5')

# model.compile(loss = 'sparse_categorical_crossentropy',
#               optimizer = 'adam',
#               metrics = ['acc'])

# predict
# x_pred = os.listdir('D:/teamproject/testset/testset')
prediction = model.predict(x)

number = np.argmax(prediction, axis = 1)

# 카테고리 불러오기
categories = ['Bichon_frise', 'Border_collie', 'Bulldog', 'Chihuahua',
              'Corgi', 'Dachshund', 'Golden_retriever', 'huskey',
              'Jindo_dog', 'Maltese', 'Pug', 'Yorkshire_terrier',
              'Doberman', 'Italian_greyhound','Pekingese', 'Sichu']

# img_path = 'D:/test/corgi.jpg'
# filename = os.listdir(img_path)


for i in range(len(number)):
    idex = number[i]
    # true = filename[i].replace('.jpg', '').replace('.png','')
    pred = categories[idex]
    print('실제 :', 'corgi', '\t예측 견종 :', pred)
    