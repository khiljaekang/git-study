'''
# 1	StandardScaler	기본 스케일. 평균과 표준편차 사용
# 2	MinMaxScaler	최대/최소값이 각각 1, 0이 되도록 스케일링
# 3	MaxAbsScaler	최대절대값과 0이 각각 1, 0이 되도록 스케일링
# 4	RobustScaler	중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화

##standardscaler :평균을 제거하고 데이터를 단위 분산으로 조정한다. 
#그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.

##mixmaxscaler: 모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
#즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다.
#특정값에 집중되어 있는 데이터가 그렇지 않은 데이터 분포보다 1표준편차에 의한 스케일 변화값이 커지게 된다.
# 한쪽으로 쏠림 현상이 있는 데이터 분포는 형태가 거의 유지된채 범위값이 조절되는 결과를 보인다.


'''
'''
####      minmaxscaler (nomalization)
 Data pre processing(데이터 전처리 과정) 중 결측값, 이상치 처리를 비롯하여 실행하는 과정이다.
#Normalization(졍규화) 또는 Standardization(표준화) 라고 한다. 이는, 0-1 사이로 전체적인 scale을 맞춰준다.

#표준화를 하지 않아도 머신러닝이 가능하지만, 값의 범위가 커짐에 따라서 학습이 정상적으로 이루어지지 않는 경우가 있다.

(범위가 더 큰 변수의 영향을 더 받게 된다.) 이에 데이터를 표준화 또는 정규화를 이용해서 값의 범위를 조절하여 사용한다.

즉, 데이터의 값이 0-1 범위내에 존재하도록 비율적으로 축소시키는 작업을 의미한다
'''






