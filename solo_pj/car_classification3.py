from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

### predict 이미지 불러오기
caltech_dir = './data/project/test_logoimage'

image_w = 112
image_h = 112

### pred 이미지를 Data 변환
X = []
filenames = []

files = glob.glob(caltech_dir + '/*.*')
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

x_pred = np.array(X)

### modelcheckpint Load
model = load_model('D:/Study/data/project/model_save/best.hdf5')

### 예측
y_pred = model.predict(x_pred)
np.set_printoptions(formatter={'float': lambda x: '{0:0.0f}'.format(x)})
cnt = 0

for i in y_pred:
    pre_ans = i.argmax()
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = '( Audi )'
    elif pre_ans == 1: pre_ans_str = '( Benz )'
    elif pre_ans == 2: pre_ans_str = '( BMW )'
    elif pre_ans == 3: pre_ans_str = '( Chevolet )'
    elif pre_ans == 4: pre_ans_str = '( Honda )'
    elif pre_ans == 5: pre_ans_str = '( Hyundai )'
    elif pre_ans == 6: pre_ans_str = '( Kia )'
    elif pre_ans == 7: pre_ans_str = '( Lexus )'
    elif pre_ans == 8: pre_ans_str = '( Toyota )' 
    else: pre_ans_str = '( Volvo )'
    if i[0] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[1] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[2] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[3] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[4] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[5] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[6] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[7] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[8] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')
    if i[9] >= 0.1 : print('해당 ' + filenames[cnt].split('\\')[1] + ' 이미지는 ' + pre_ans_str + ' 로 추정됩니다.')


    cnt += 1
