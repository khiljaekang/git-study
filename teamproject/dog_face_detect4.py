import dlib, cv2, os                            # os.walk뺀거
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()  #시작하는 시간을 알려준다. 

train_path = 'D:\\breed'  #train path = 를 우리가 넣어둔 3종류의 개의 사진이 있다. 

def face_detector(path, folder, w, h):       #Face detector 이라는 함수에는 path, folder, w, h를 넣어줄테니 그것들을 가지고 아래와 같이 계산해다. 
    print('---------- START %s ---------'%(folder)) #시작하고 난 다음에 %의 자리에 folder의 명을 입력해준다. 
    image_dir = path + '/'+folder+'/'               #image_dir는 path + '/'+folder+'/' 라는것 즉 파일의 경로를 지정해주겠다.       
    X = []                                          #x의 빈 리스트를 만들어준다. 

    f = os.listdir(image_dir)                       #아까 위에서 지정해준 image_dir를 가지고 listdir 를 해주면서 폴더내 파일 이름을 가져와준다. 

    for filename in f:                  # 파일 별로 이미지 불러오기#filename이라는 것을 지정해주고 각각의 파일들을 f에서 가져와주겠다는 것이다. 
        img = cv2.imread(image_dir + filename)    #image_dir = path/folder/로 되어있는 경로이다 여기다가 filename은 chihuahua, retriever와 같은 개들 파일이름을 가져와주게 된다. 
                                                  #즉 image_dir+filename = path/folder/chihuahua와 같은 파일의 경로를 만들어주고 이것을 for문을 돌렸으니 각각의 견종마다의 파일들을 다 불러와주게 되는 것이다. 
                                                  #imread는 뒤에 적어둔 경로에서 그림파일들을 읽어오게 해주는 것이다. 지금 상태에서는 각각의 견종폴더들에서 사진을 불러와 주겠지. 
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv는 BGR로 불러 들임으로 볼 때 우리가 원하는 색으로 보기 위해 RGB로

        img_result = img.copy()                     # 원본 이미지를 copy 해주겠다는 의미이고 copy 를 해주는 이유는 원본에 영향을 주지 않게 하기 위해서이다. 

        detector = dlib.cnn_face_detection_model_v1('D:/weight\dogHeadDetector.dat') #detector에는 dlib에서 이미 만들어진 가중치를 가져오면된다. 
        dets = detector(img, upsample_num_times=1)  #dets는 detector이라는 가중치에서 우리가 이미 불러온 개사진을 넣어주게 된다.  

        x = X.append

        for i, d in enumerate(dets):   #enumerate를 사용하게 된다면 우리가 가지고 있는 파일들에 넘버링을 해주게 된다. 

            x1, y1 = d.rect.left(), d.rect.top()
            x2, y2 = d.rect.right(), d.rect.bottom()
            pad = (x2 - x1)
            #pad 를 사용한 이유는 강아지 얼굴을 너무나도 타이트하게 잡아버리기 때문에 인위적으로 바운딩 박스를 늘려준 것이다. 
            x1 = x1 - pad/4
            y1 = y1 - pad*3/8
            x2 = x2 + pad/4
            y2 = y2 + pad/8
            #개들마다 바운딩 박스의 크기가 각각 다르니까 박스의 비율로 해주기 위해서 가로축의 길이를 pad라고 지정해준것이다. 
            #가로축으로 해 준 이유는 가로 새로의 길이의 차이가 얼마 나지 않아서 일부러 그렇게 잡았다. 
            #y축은 위쪽 머리는 잘 인식하는데 하관은 인식을 잘 하지 못해서 
            x1, x2, y1, y2 = map(int, (x1, x2, y1, y2)) # int형으로 변환
            # 정수형밖에 못 들어가니까 우리가 가지고 있는 것들은 소수가 있을수도 있기에 int형으로 바꾸어 주게된다. 
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            #강아지의 얼굴이 항상 가운데만 있는게 아니라 끝0,0의 좌표에 위치해 있을때는 음의 값으로 넘어가지 않게.        
            cropping = img[y1:y2, x1:x2]
            crop = cv2.resize(cropping, dsize = (w, h), interpolation = cv2.INTER_LINEAR)
            #cropping은 인덱싱 하는것 처럼 크롭 해준다고 생각 해주면 된다. 원하는 사이즈로 리사이즈 해준다. 
            #interpolation 은 리사이징 와중에 비게 되는 곳들을 채워주기 위해서 사용해주는 기법으로는 interpolation의 선형 보간법이다. 
            x(crop/255)
        
        ''' 견종별로 따로 따로 저장 '''
    images = np.array(X)

    np.save('D:\data/numpy/face_image_%s.npy'%(folder), images)

    print('---------- END %s ---------'%(folder))

    
face_detector(train_path, 'Retriever', 128,128)




