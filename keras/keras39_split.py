import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 11)) # 1~ 10 : 
size = 5

def split_x(seq, size):
    aaa = []        # 는 리스트
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) # item for item in subset = 굳이 안넣고 subset 하면 간단.
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("==============")
print(dataset)

# split_x(seq, size) 아래보면, dataset = split_x(a, size) 로 정의되었고
# a = seq임을 확인,
# for i in range(6) : range(6) = 0,1,2,3,4,5 = i
# subset = seq[0:(0+5)] = seq[0:5] = 0 ~ 4 = 1,2,3,4,5
# aaa.append([item]) = 1,2,3,4,5 = aaa
# aaa = 12345, 

# 1:6 = 1~5 = 2,3,4,5,6
# 2:7 = 2~6 = 3,4,5,6,7
# 3:8 = 3~7 = 4,5,6,7,8
# 4:9 = 4~8 = 5,6,7,8,9
# 5:10 = 5~9 = 6,7,8,9,10


# def split_x(seq, size):
#     aaa = []
#     for i in range(len(a) - size + 1):
#         subset = a[i : (i + size)]
#         print(subset)
#         # aaa.append([item for item in subset])
#         aaa.append(subset)
#     print(type(aaa))
#     return np.array(aaa)