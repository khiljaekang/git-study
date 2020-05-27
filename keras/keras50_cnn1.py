from keras.models import Sequential
from keras.layers import Conv2D      
from keras.layers import MaxPooling2D, Dense, Flatten
#2.모델구성
model = Sequential()
model.add(Conv2D(10,(2,2),  input_shape=(10,10,1)))  #(9, 9, 10)
           #conv해서 나가는 값이 10개다 , 가로세로 (2,2) 
           # (10,10,1) = 가로,세로,색깔   1= 흑백
           # ex)10000장 일 경우 x = (10000, 10, 10, 1)
           # 행 무시 되기 때문에 10,10,1
           #padding: 경계 처리 방법을 정의한다. 'valid' = 유효한 영역만 출력되며,입력 사이즈보다 작다.
           #padding의 default값은 valid
           #'same'은 출력 이미지 사이즈가 입력 이미지 사이즈와 동일.
           #(10, (2, 2))    = 가로세로 2로 자르겠다 = kernel_size =2
           #filter, kernel_size 
           #input_shape=(10,10,1)
           #height, width, chanel (batch_size) 4차원(행,가로,세로,색깔)-행무시 
           #stride가 default 값으로 숨어 있고 default값은 1이다.
           #stride는 몇칸씩 묶을 것인가를 결정한다. 
           #maxpooling = 필요한 데이터를 날리고 중요한 부분만 추출하겠다.ex) (4,4)→(2,2)
model.add(Conv2D(7, (3,3), ))               #(7, 7, `7)
model.add(Conv2D(5, (2,2), padding='same')) #(7, 7, 7)
model.add(Conv2D(5, (2,2)))                #(6, 6, 5)
model.add(Conv2D(5, (2,2), strides=2 ))                #(3, 3, 5)
model.add(Conv2D(5, (2,2), strides=2, padding='same'))  #(2, 2, 5)
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten()) # 그 윗단에 있는 레이어를 쫙펴준다 Dense형으로 바꿔줌.
model.add(Dense(1))  #conv레이어의 끝은 항상 flatten 이다

model.summary()
  
##CNN_parameter


#  :  ( channel * kernel_size  * filter ) + ( bias * filter)

#  ex) (   1    *   (2 * 2)    *   10   ) + (  1   *   10  )
