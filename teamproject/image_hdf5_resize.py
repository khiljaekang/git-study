import cv2
import numpy as np
import h5py

# image

dataset = np.load('D:/data/face_image_total.npy')

def resize(dataset, w, h):

    f = h5py.File('D:/data/image_hdf5.hdf5','a')

    imageset = f.create_dataset('imageset%d'%(w), (dataset.shape[0], h, w, 3)) #, dtype='float64')

    for i in range(len(dataset)):
        print(i)
        img = dataset[i]
        img_umat = cv2.UMat(img)            ## opencv opencl-gpu ro dol ril su it ge sok dodo PPal la jim
                
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
    f.create_dataset('label%d'%(w), data = label)
    print(f['label%d'%(w)][:].shape[0])
'''