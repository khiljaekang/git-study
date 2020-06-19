from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt 
import glob
import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

np.random.seed(15)

path = './data/project/logos sample'

# create image data augmentation generator

datagen = ImageDataGenerator(zoom_range = 0.5,
                             rescale = 1. / 255, 
                             height_shift_range = 0.1,
                             width_shift_range = 0.1,
                            )
i = 0

for train_gen in datagen.flow_from_directory( path,
                                         target_size = (150, 150),
                                         batch_size = 25,
                                         class_mode = 'categorical',
                                         save_to_dir= './data/project/generator',     # 저장 경로
                                         save_prefix= 'image',                        # 파일명
                                         save_format = 'jpg'                          # 파일 형식
                                         ):                        
    i += 1
    if i > 25:
        break
