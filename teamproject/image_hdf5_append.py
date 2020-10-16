import h5py
'''
t = h5py.File('D:/data/image_hdf5.hdf5', 'r')
t = t['imageset512'][:]
print(t.shape)

d = h5py.File('D:/data/hdf5/face_Doberman.hdf5', 'r')
d = d['doberman'][:431]

i1 = h5py.File('D:/data/hdf5/face_Italian_Greyhound.hdf5', 'r')
i1 = i1['Italian_Greyhound'][:390]

i2 = h5py.File('D:/data/hdf5/face_Italian_Greyhound_part2.hdf5', 'r')
i2 = i2['Italian_Greyhound'][:109]

p = h5py.File('D:/data/hdf5/face_Pekingese.hdf5', 'r')
p = p['pekingese'][:463]

s = h5py.File('D:/data/hdf5/face_Sichu.hdf5', 'r')
s = s['Sichu'][:558]



with h5py.File('D:\data\hdf5/DATA.hdf5','a') as f:
    imageset = f.create_dataset('512', (7481, 512, 512, 3), maxshape = (None, 512, 512, 3))

    start = 0

    for i, x in zip([5530, 431, 390, 109, 463, 558],[t, d, i1, i2, p, s]):
        end = start + i
        print(end)

        imageset[start:end] = x

        start = end

    print('------------ complete -----------')
'''
t = h5py.File('D:/data/image_hdf5.hdf5', 'r')
t = t['label1024'][:]
print(t.shape)

with h5py.File('D:\data\hdf5/DATA.hdf5','a') as f:
    del f['label']
    label = f.create_dataset('label', (7481, ), maxshape = (None, ))
    start = 0
    k = 12

    for i in [5530, 431, 390+109, 463, 558]:
        end = start + i
        print(end)

        if i == 5530:
            label[start:end] = t
        else:
            label[start:end] = k
            k += 1

        start = end

    print('------------ complete -----------')