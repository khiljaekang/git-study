import numpy as np
import cv2
import re
import os
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model
# data
#------------- only one file -------------------
path = 'D:/teamproject/testset/testset/retriever.png'
img = cv2.imread(path)
img = cv2.resize(img, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)
x_pred = np.array(img/255).reshape(-1, 256, 256, 3)                 # 픽셀 = 0~255값가짐, 학습을 위해 0~1사이의 소수로 변환
#---------------- load_data -------------------
# x_pred = np.load('./project/project02/data/pred_img.npy')
# path = 'D:/data/project/testset'                                  # filename


# load_model
model = load_model('D:/teamproject/checkpoint/efficientnet_true2.hdf5')

# predict
prediction = model.predict(x_pred)
number = np.argmax(prediction, axis = 1)

# 카테고리 불러오기
categories = np.load('D:/teamproject/data/category.npy')
print(categories)
# filename = os.listdir(path)
# filename = ['retriever']

# for i in range(len(number)):
#     idex = number[i]
#     true = filename[i].replace('.jpg', '').replace('.png','')
#     pred = categories[idex]
#     print('실제 :', true, '\t예측 견종 :', pred)

# 'Bichon_frise' 'Border_collie' 'Bulldog' 'Chihuahua' 'Corgi' 'Dachshund'
#  'Golden_retriever' 'huskey' 'Jindo_dog' 'Maltese' 'Pug'   
#  'Yorkshire_terrier'