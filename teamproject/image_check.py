import h5py
import cv2

with h5py.File('D:/teamproject/data/pekingese.hdf5', 'r') as f:
    img = f['pekingese'][145]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', img)
    cv2.waitKey(0)