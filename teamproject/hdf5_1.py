import cv2
import numpy as np
import h5py

path = 'D:/face'

def resize(path, w, h):

    f = h5py.File('D:/teamproject/data/70400.hdf5', 'a')    # 'w' : write = 없으면 생성. 있으면 리셋 후 재생성
                                                    # 'r' : read  = 읽기 전용
                                                    # 'a' :       = 없으면 생성, 있으면 갱신  

    imageset = f.create_dataset('70400', (70400, w, h, 3)) #, dtype='float64')

    for i in range(len(dataset)):
        print(i)
        img = dataset[i]
        img_umat = cv2.UMat(img)                 
        img_result = cv2.resize(img_umat, (w, h), interpolation = cv2.INTER_LINEAR)
        img_result = cv2.UMat.get(img_result)       ## UMat.get = numpy ro ba QUim

        imageset[i] = img_result    # hdf5로 저장

    f.close()

    return 

resize(dataset, 256, 256)
'''
# label
label = np.load('D:/data/face_label_total.npy')
w = 1024

with h5py.File('D:/data/image_hdf5.hdf5','a') as f:
    f.create_dataset('label', data = label)
    print(f['label'][:].shape[0])
'''