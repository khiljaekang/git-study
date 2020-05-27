'''
# 1. 사이킷 런 개요

# 머신러닝 모델을 학습하고 그 결과를 검증하기 위해서는 원래의 데이터를 Training, Validation, Testing의
# 용도로 나누어 다뤄야 한다. 그렇지 않고 Training에 사용한 데이터를 검증용으로 사용하면 시험문제를 알고
# 있는 상태에서 공부를 하고 그 지식을 바탕으로 시험을 치루는 꼴이므로 제대로 된 검증이 이루어지지 않기
# 때문이다. 

# 2. Parameter & Return

#from sklearn.model_selection import train_test_split
#train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)

# Parameter

# arrays : 분할시킬 데이터를 입력 (Python list, Numpy array, Pandas dataframe 등..)

# test_size : 테스트 데이터셋의 비율(float)이나 갯수(int) (default = 0.25)

# train_size : 학습 데이터셋의 비율(float)이나 갯수(int) (default = test_size의 나머지)

# random_state : 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값 (int나 RandomState로 입력)

# shuffle : 셔플여부설정 (default = True)

# stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진
#  Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.

# Return

# X_train, X_test, Y_train, Y_test : arrays에 데이터와 레이블을 둘 다 넣었을 경우의 반환이며, 데이터와 레이블의 순서쌍은 유지된다.

# X_train, X_test : arrays에 레이블 없이 데이터만 넣었을 경우의 반환


####사이킷런###
사이킷런 파이썬 프레임 워크는 탄탄한 학습 알고리즘이 장점
설치,학습,사용하기 쉽고 예제와 사용 설명서가 잘 돼있음
딥러닝이나 강화학습을 다루지 않는 단점
그래픽 모델과 시퀀스 예측 기능을 지원하지 않음
파이썬 이외의 언어에서는 사용할 수 없고, 파이썬 JIT 컴하일러인 파이파이나 GPU를 지원하지 않음
사이킷런은 분류와 회귀, 클러스터링,차원 축소,모델 선택, 전 처리에 대해 다양한 알고리즘을 지원,
이와 관련된 문서와 예제도 훌륭하다
하지만 이런 작업을 완료하기 위한 안내 워크플로우가 전혀 없음
딥러닝이나 강화 학습을 지원하지 않아 정확한 이미지 분류와 신뢰성 있는 실시간 언어 구문 분석, 번역 같은 문제를
해결하는 데는 적절치 않음
여러 가지 다른 관측값을 연결하는 예측 함수를 만드는 것부터 관측값을 분류하는 것,라벨이 붙어있지 않은 데이터
세트의 구조를 학습하는 것까지, 수 십개의 뉴런 계층이 필요 없는 일반적인 머신러닝 용도라면 사이킷만한 것이 없음

'''