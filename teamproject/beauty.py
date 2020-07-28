import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np

# Load Models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('D:/study/model/shape_predictor_5_face_landmarks.dat')

# Load images
img = dlib.load_rgb_image('D:/study/faceimages/images/junghyun.png')

# plt.figure(figsize=(16, 10))
# plt.imshow(img)
# plt.show()

# Find face

img_result = img.copy()

dets = detector(img, 1)

if len(dets) == 0:
    print('cannot find faces!')

fig, ax = plt.subplots(1, figsize=(16, 10))

for det in dets:
    x, y, w, h = det.left(), det.top(), det.width(), det.height()

    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# ax.imshow(img_result)
# plt.show()

# Find Landmarks 5points
# fig, ax = plt.subplots(1, figsize=(16,10))

objs = dlib.full_object_detections()

for detection in dets:
    s = sp(img, detection)         # s = shape
    objs.append(s)

    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)

ax.imshow(img_result)
plt.show()

# Align Faces

# faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)

# fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))

# axes[0].imshow(img)

# for i, face in enumerate(faces):
#     axes[i+1].imshow(face)

# plt.show()

#functionalize

def align_faces(img):
    dets = detector(img,1)

    objs = dlib.full_object_detections()

    for detection in dets:
        s = sp(img, detection)
        objs.append(s)

    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)

    return faces

#Load BeautyGAN Pretraind
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('D:/study/model/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('D:/study/model/'))
graph = tf.get_default_graph()

X = graph.get_tensor_by_name('X:0')  #source 노 매이크업 이미지
Y = graph.get_tensor_by_name('Y:0')  #reference 매이크업을 따라 할 이미지
Xs = graph.get_tensor_by_name('generator/xs:0')   #output 제너레이터가 만들 아웃풋의 이미지

def preprocess(img):
    return img.astype(np.float32) / 127.5 - 1.            #전처리 0~255 를 -1에서 1사이에 값을 가진 플롯형태로 바꿈

def postprocess(img):
    return ((img + 1.) * 127.5).astype(np.uint8)          #후처리  전처리의 반대

#load Images

img1 = dlib.load_rgb_image('D:/study/faceimages/images/junghyun.png')
img1_faces = align_faces(img1)

img2 =dlib.load_rgb_image('D:/study/faceimages/makeup/licEnH3rBjSA.png')
img2_faces = align_faces(img2)

# fig, axes = plt.subplots(1, 2, figsize=(16, 10))
# axes[0].imshow(img1_faces[0])
# axes[1].imshow(img2_faces[0])
# plt.show()

#Run

src_img = img1_faces[0]     
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)


output = sess.run(Xs, feed_dict={
    X: X_img,
    Y: Y_img

})

output_img = postprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()
  

  





