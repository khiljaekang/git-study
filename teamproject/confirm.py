from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
import numpy as np
import cv2
import matplotlib.pyplot as plt

x = np.load('D:/teamproject/data/face_image_Yorkshire_terrier.npy')
print(x.shape)


image = x[2].astype(np.float32)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('image',img)
cv2.waitKey(0)
