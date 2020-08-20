import numpy as np
import cv2                          
import glob
import os
import re                                 
import h5py
''' 단순 이미지 불러오기 
    os.walk 사용 X'''

path = 'D:/teamproject/face'      # dataset 상위 폴더 경로

# 이미지 불러오기
def load_image(path, w, h):   # 폴더별로 이미지 불러오고 labeling
    folder_name = os.listdir(path) 
    
    # resize
    image_w = w     # width
    image_h = h     # height

    X = []

    hdf = h5py.File('D:/teamproject/data/image_kface_front.hdf5', 'a')
    del hdf['image_kface_front']
    imageset = hdf.create_dataset('image_kface_front', (400, w, h, 3), maxshape=(None, w, h, 3))

    i = 0
    
    for folder in folder_name:
        
        image_dir = path + '/'+ folder + '/'  #불러올 이미지 경로

        f = os.listdir(image_dir)
        
        for filename in f:  
            print(image_dir+filename)
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, dsize = (image_w, image_h), interpolation = cv2.INTER_LINEAR)

            print(img.shape)
            img = img.reshape(-1, h, w, 3)

            imageset[i] = img

            i += 1
            
    return 

load_image(path, 256, 256)

print('-----Data Save Complete------')
