import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from PIL import Image
import os, glob
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from keras.preprocessing.image import img_to_array

caltech_dir = './data/project/logos'
brands =['hyundai logo', 'kia logo',]
nb_classes = len(brands)             #nb_classes 개수 정의

print
image_w = 150
image_h = 150

# pixels = image_h * image_w * 3          #pixels 색 정의

X = []
Y = []
'''
리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능
enumerate는 “열거하다”라는 뜻이다. 이 함수는 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다.
보통 enumerate 함수는 아래 예제처럼 for문과 함께 자주 사용된다
'''

X = []
Y = []

for idx, brand in enumerate(brands):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    print(idx)
    
    image_dir = caltech_dir + '/' + brand  ### image_dir 내 하위 디렉토리(label)를 가져온다
    files = glob.glob(image_dir + "/*.*")  ### caltech_dir 는 이미지 경로
    print(brand, " 파일 길이 : ", len(files)) ###glob 모듈에 대해 알아보도록 하죠. glob는 파일들의 목록을 뽑을 때 사용
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

        if i % 700 == 0:
            print(brand, ':', f)
            
x = np.array(X)
y = np.array(Y)

print(x.shape) #(5500, 150, 150, 3)
print(y.shape) #(5500, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
xy = (x_train, x_test, y_train, y_test)
np.save('./data/project/logos.npy', xy)
print(x_train.shape)   #(4400, 150, 150, 3)
print(x_test.shape)    #(1100, 150, 150, 3)
print(y_train.shape)   #(4400, 10)
print(y_test.shape)    #(1100, 10)


