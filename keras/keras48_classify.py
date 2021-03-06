
# """train_test_predict분리하지 말고"""
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense

# #1. 데이터
# x = np.array(range(1, 11))
# y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) 

# print(x.shape) # (10,) 스칼라가 10개 벡터가 1개 = input_dim=1 
# print(y.shape) # (10,)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(100, input_dim = 1, activation='relu')) 
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) #output에 activation='sigmoid'


# #3. 컴파일, 훈련
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
# model.fit(x, y, epochs = 100, batch_size = 1, )

# #분류모델에서는 metrics를 accuracy를 쓴다. 
# #2진분류에서 loss 값은 'binary_crossentropy'


# #4. 평가, 예측
# loss, acc = model.evaluate(x, y, batch_size= 1)
# print('loss :', loss)
# print('acc :', acc)

# x_pred = np.array([1,2,3])
# y_pred = model.predict(x_pred)
# print('y_pred :', y_pred)

# y_predict = model.predict(x)
# print(y_predict)

'''
#################과제##################
 0과 1이 나오게 만들어라
 1.잘만들어온 걸 퍼오던지
 2.sigmoid를 함수로 불러오던지
 3.기타등등
'''

# '''20200526 분류모델'''

# 회귀모델을 먼저 구현해보자.
# 1. 모듈 임포트
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
import numpy as np

# 1-1. 조기종료 객체 생성
es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 10)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 2. 데이터
x = np.array(range(1, 11))
y = np.array([1, 0 ,1, 0, 1, 0, 1, 0, 1, 0])
y = sigmoid(y)
print(x.shape)
print(y.shape)

# 3. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape = (1, ), activation = 'relu'))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1, activation = 'sigmoid'))
'''
분류모형의 활성화 함수 - 마지막 아웃풋 레이어에 추가
1. sigmoid
2. hard_sigmoid
3. softmax
'''
model.summary()

# 4. 실행 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x, y, epochs = 1000,
          batch_size = 1, callbacks = [es])
'''
분류모형의 손실함수
1. Cross-Entropy Loss
2. Categorical Cross-Entropy Loss
3. Binary Cross-Entropy Loss        - 이진분류 모델은 딱 이거 하나 !!
4. Focal loss (함수로 구현해서 사용)
'''

# 5. 평가 및 예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print("loss : ", loss)
print("acc : ", acc)

x_pred = np.array([1, 2, 3])
pred = model.predict(x_pred).reshape(3, )
print("pred : \n", pred)


# 기본적인 분류모델
# 결과치가 2가지로만 나오는 모델
# 이진 분류 모델 - Binary Classification Model
# activation 활성화 함수 - sigmoid
# 분류모형의 손실함수
# 1. Cross-Entropy Loss
# 2. Categorical Cross-Entropy Loss
# 3. Binary Cross-Entropy Loss
# 4. Focal loss (함수로 구현해서 사용)
# def focal_loss(gamma = 2., alpha = .25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))
#                         - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed

'''
pred :
 [[0.5392872]
  [0.5209384]
  [0.5024807]]     # 0.5를 기준으로 0, 1로 분류
'''
# tmp = []
# a = np.round(pred[0][0])
# b = np.round(pred[1][0])
# c = np.round(pred[2][0])
# print(a)
# print(b)
# print(c)
# tmp.append([a, b, c])
# print(tmp)

for i in range(len(pred)):
    if i >= 0.5:
        pred[i] = 1
    else:
        pred[i] = 0
print(pred)

