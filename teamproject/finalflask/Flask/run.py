# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, jsonify, url_for
from flask import current_app, send_from_directory, send_file
from werkzeug.utils import secure_filename
from imutils import face_utils
import datetime
import os
import cv2
import dlib
import glob
import tensorflow as tf
from tensorflow.keras.models import load_model 
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import h5py
import numpy as np
import keras
from keras.models import model_from_json

from matplotlib import pyplot
from numpy import linspace

from AdaIN import AdaInstanceNormalization
import keras.backend.tensorflow_backend as tb 
tb._SYMBOLIC_SCOPE.value = True


os.environ["CUDA_VISIBLE_DEVICES"]='-1'


app = Flask(__name__, static_url_path = "", static_folder = "")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
categories = ['Bichon_frise', 'Border_collie', 'Bulldog', 'Chihuahua',
              'Corgi', 'Dachshund', 'Golden_retriever', 'huskey',
              'Jindo_dog', 'Maltese', 'Pug', 'Yorkshire_terrier',
              'Doberman', 'Italian_greyhound','Pekingese', 'Sichu']

label_dict = {}
for i in range(len(categories)):
    label_dict[i] = categories[i]
model = None
detector = dlib.cnn_face_detection_model_v1('model/dogHeadDetector.dat')
predictor = dlib.shape_predictor('model/landmarkDetector.dat')

def load_dog_model():
    global model
    model = load_model('model/efficientnet_true2.hdf5')


def category_model(new_file_name):
    print(new_file_name)
    
    

    img = cv2.imread('_uploads/{}'.format(new_file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img, upsample_num_times=1)
    img_result = img.copy()
    for i, d in enumerate(dets):
        print("Detection {}: Left:{} Top:{} Right:{} Bottom:{} Confidence:{}".format(i, 
            d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        s = 30
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()

        cv2.rectangle(img_result, (x1, y1-s), (x2+s, y2+s), thickness=2, color=(122, 122, 122), lineType=cv2.LINE_AA)
    shapes = []
    




    for i, d in  enumerate(dets):
        shape = predictor(img, d.rect)
        shape = face_utils.shape_to_np(shape)

        for i, p in enumerate(shape):
            shapes.append(shape)
            cv2.circle(img_result, center=tuple(p), radius=3, color=(0,0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        print('2')

    img_out = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('_dog_face/%s'%(new_file_name), img_out)


    img = cv2.imread('_uploads/{}'.format(new_file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_result = img.copy()                     # 원본 이미지 copy
    dets = detector(img, upsample_num_times=1)

    for i, d in enumerate(dets):
        
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()
        pad = (x2 - x1)
            
        #---------------- bbox 키우기 ----------------
        x1 = x1 - pad/4
        y1 = y1 - pad*3/8
        x2 = x2 + pad/4
        y2 = y2 + pad/8
        
        x1, x2, y1, y2 = map(int, (x1, x2, y1, y2)) # int형으로 변환
        # print('가로 : ',x1, x2)
        # print('세로 : ',y1, y2)                    
        cv2.rectangle(img_result, (x1, y1), (x2, y2), thickness=1, color=(122, 122, 122), lineType=cv2.LINE_AA)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
            
        cropping = img[y1:y2, x1:x2]
        crop = cv2.resize(cropping, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)
        

        img_result = (crop)



    # img = cv2.resize(img, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)
    img = img_result.reshape(-1, 256, 256, 3)
    x = img.astype('float')/255
    
    if model ==None:
        load_dog_model()

    prediction = model.predict(x)

    per = np.max(prediction, axis = 1)

    prediction = (np.argmax(prediction, axis = 1))[:]
    
    # print('예측값 :', label_dict[int(prediction)], prediction, 'Percentage :', per*100)
    
    dog_predict = label_dict[int(prediction)]
    percentage = per * 100
    return (dog_predict, percentage)




def loadModel(name): #Load a Model
        
        file = open("./model/"+"gen_c13"+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
        mod.load_weights("./model/"+name+".h5")
        
        return mod
    

def load(): #Load JSON and Weights from /Models/
    
    gen = loadModel("gen_c13_464")

    return gen


def noise():
    return np.random.normal(0.0, 1.0, size = [1, 512])








def plot_vector(examples,n, file_name):
    for i in range(n):
        pyplot.subplot(1,n,i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i].reshape(256,256,3))
    # plt.tight_layout()
    
    plt.savefig("./_dogvector/{}".format(file_name),dpi=1028,bbox_inches='tight')
    # plt.show()

def dog_vector(img,p,p2,model, file_name):

    examples = []

    ratios = linspace(0,1,10)

    for i in ratios:
        
        noise = (1 - i) * p + i * p2
        examples.append(model.predict([noise, img, np.ones((1,1))]))
        


    plot_vector(examples,10,file_name)

    
    



def gan_model(new_file_name):

    gen = load()
    img = cv2.imread('_uploadsgan/{}'.format(new_file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize = (256, 256), interpolation = cv2.INTER_LINEAR)
    noiseimage = img.reshape(1,256,256,-1)/255
    noisew = noise()
    noisew2 = noise()
    dog_vector(noiseimage , noisew, noisew2, model=gen, file_name=new_file_name)



# 파일 업로드 용량 제한하기 아래는 16MB가 최대 제한
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/gan')
def gan():
    return render_template('gan.html')





@app.route('/upload_low', methods=['GET', 'POST'])

def upload_low():
    import os
    print(os.getcwd())
    print(os.path.realpath(__file__))
    print('upload_low....')
    if request.method == 'POST':
        print("Posted file: {}".format(request.files['file']))
        f = request.files['file']
        new_file_name = secure_filename(f.filename)

        model_type = request.form['type']

        if '.' in new_file_name:
            file_name, file_extension = os.path.splitext(new_file_name)
            file_extension = file_extension[1:]
        else:
            file_extension = new_file_name

        file_name = str(datetime.datetime.now())
        new_file_name = file_name + '.' + file_extension
        new_file_name = secure_filename(new_file_name)
        f.save('_uploads/' + new_file_name)

        # 이미지 생성하기 >> 품종
        dog_pre, per =category_model(new_file_name)
        # file_name = generate(new_file_name, model_type)
        file_name = new_file_name
        # 생성된 이미지 파일명을 리턴하면 index.html에서 해당 이미지 표시 및 다운 버튼을 생성
        return file_name +"/"+str(dog_pre)+"/"+str(float(per))+"%"


@app.route('/download_gen/<path:file_name>', methods=['GET', 'POST'])
def download_gen(file_name):
    return send_from_directory('_uploads', file_name, as_attachment=True)


@app.route('/upload_gan', methods=['GET', 'POST'])

def upload_gan():
    import os
    print(os.getcwd())
    print(os.path.realpath(__file__))
    print('upload_gan....')
    if request.method == 'POST':
        print("Posted file: {}".format(request.files['file']))
        f = request.files['file']
        new_file_name = secure_filename(f.filename)

        model_type = request.form['type']

        if '.' in new_file_name:
            file_name, file_extension = os.path.splitext(new_file_name)
            file_extension = file_extension[1:]
        else:
            file_extension = new_file_name

        file_name = str(datetime.datetime.now())
        new_file_name = file_name + '.' + file_extension
        new_file_name = secure_filename(new_file_name)
        f.save('_uploadsgan/' + new_file_name)

        # 이미지 생성하기 >> 강아지
        gan_model(new_file_name)
        # file_name = generate(new_file_name, model_type)
        file_name = new_file_name
        # 생성된 이미지 파일명을 리턴하면 index.html에서 해당 이미지 표시 및 다운 버튼을 생성
        return file_name

@app.route('/download_gen2/<path:file_name>', methods=['GET', 'POST'])
def download_gen2(file_name):
    return send_from_directory('_dogvector', file_name, as_attachment=True)



if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=7777, debug=False, threaded= False)