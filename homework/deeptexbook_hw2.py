
##############################14.1.1 pandas로 데이터 읽기 ##############################
import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header =None)
# 각 수치가 무엇을 나타내는지 컬럼 헤더를 추가합니다.
df.columns =[ "", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Mangnesium", 
            "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", 
            "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

print(df)
#         Alcohol  ...  OD280/OD315 of diluted wines  Proline
# 0    1    14.23  ...                          3.92     1065
# 1    1    13.20  ...                          3.40     1050
# 2    1    13.16  ...                          3.17     1185
# 3    1    14.37  ...                          3.45     1480
# 4    1    13.24  ...                          2.93      735
# ..  ..      ...  ...                           ...      ...
# 173  3    13.71  ...                          1.74      740
# 174  3    13.40  ...                          1.56      750
# 175  3    13.27  ...                          1.56      835
# 176  3    13.17  ...                          1.62      840
# 177  3    14.13  ...                          1.60      560

# [178 rows x 14 columns]


# 문제
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header =None)
df.colums = ["sepal length", "sepal width", "petal length", "petal width", 'class']
print(df)

##############################14.1.2 CSV라이브러리로 CSV만들기 ##############################
import csv

# with 문을 사용해서 파일을 처리
with open("csv0.csv", "w")as csvfile:
    # writer() 메서드의 인수로 csvfile과 개행 코드(\n)를 지정
    writer = csv.writer(csvfile, lineterminator="\n")

    # writerow(리스트)로 행을 추가
    writer.writerow(["city", "year", 'season'])
    writer.writerow(["Nagano", "1998", 'winter'])
    writer.writerow(["Sydney", "2000", 'summer'])
    writer.writerow(["Salt Lake city", "2002", 'winter'])
    writer.writerow(["Athens", "2004", 'summer'])
    writer.writerow(["Torino", "2006", 'winter'])
    writer.writerow(["Beijing", "2008", 'summer'])
    writer.writerow(["Vancouver", "2010", 'wimter'])
    writer.writerow(["London", "2012", 'summer'])
    writer.writerow(["Sorchi", "2014", 'winter'])
    writer.writerow(["Rio de Janeiro", "2016", 'summer'])
    
    
##############################14.1.3 PANDAS로 CSV만들기 ##############################    
    
    

import  pandas as pd

data = {'city': ["Nagano","Sydney","Salt Lake city","Athens","Torino"
                ,"Beijing","Vancouver","London","Sorchi","Rio de Janeiro"],
        'year':["1998",'2000','2002','2004','2006',
                '2008','2010','2012','2014','2016'],
        'season':['winter','summer','winter''summer','winter',
                  'summer','winter','summer','winter','summer']}

df = pd.DataFrame(data)

df.to_csv("csv1.csv")

# 문제
data = {'OS': ['Machintosh', 'Windows','Linux'],
        'relese': [1984, 1985, 1991],
        'country': ['US','US','']}

df = pd.DataFrame(data)
df.to_csv("OSlist.csv")


##############################14.2 DATA FRAME 복습 ##############################


import pandas as pd
from pandas import Series, DataFrame

attri_data1 = {'ID':['100','101','102','103','104','106','108','110','111','113'],
               "city": ["서울",'부산','대전','광주','서울','서울','부산','대전','광주','서울'],
               "brith_day":[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name":['영이','순돌','짱구','태양','션','유리','현아','태식','민수','호식']}

atrri_data_frame1 = DataFrame(attri_data1)
print(atrri_data_frame1)
#     ID city  brith_day name
# 0  100   서울       1990   영이
# 1  101   부산       1989   순돌
# 2  102   대전       1992   짱구
# 3  103   광주       1997   태양
# 4  104   서울       1982    션
# 5  106   서울       1991   유리
# 6  108   부산       1988   현아
# 7  110   대전       1990   태식
# 8  111   광주       1995   민수
# 9  113   서울       1981   호식


attri_data2 = {'ID': ['107','109'],
               'city':['봉화','전주'],
               'brith_day':[1994, 1988]}
attri_data_frame2 = DataFrame(attri_data2)

atrri_data_frame1.append(attri_data_frame2).sort_values(by ='ID', ascending = True).reset_index(drop=True)
print(attri_data_frame2)
#     ID city  brith_day name
# 0  100   서울       1990   영이
# 1  101   부산       1989   순돌
# 2  102   대전       1992   짱구
# 3  103   광주       1997   태양
# 4  104   서울       1982    션
# 5  106   서울       1991   유리
# 6  108   부산       1988   현아
# 7  110   대전       1990   태식
# 8  111   광주       1995   민수
# 9  113   서울       1981   호식
#     ID city  brith_day
# 0  107   봉화       1994
# 1  109   전주       1988


##############################14.3 결측치 ##############################

import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 일부 데이터를 누락 시킵니다.
sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

print(sample_data_frame)
#           0         1         2         3
# 0  0.325538  0.528041  0.634729  0.755322
# 1       NaN  0.872643  0.507700  0.333328
# 2  0.898748  0.964324       NaN  0.963705
# 3  0.178094  0.517255  0.477483  0.044298
# 4  0.600127  0.147818  0.257669  0.290597
# 5  0.337104  0.331577  0.964237       NaN
# 6  0.769419  0.426936  0.889084       NaN
# 7  0.437038  0.326573  0.232447       NaN
# 8  0.544041  0.231620  0.806235       NaN
# 9  0.923168  0.593689  0.357183       NaN


'''
# 리스트와이즈 삭제 ( listwise delection)
: 데이터가 누락된 행(NaN을 가진 행)을 통째로 지우는 것
'''
print(sample_data_frame.dropna())
#           0         1         2         3
# 0  0.639910  0.590755  0.621288  0.472937
# 3  0.391421  0.175547  0.760504  0.499116
# 4  0.786901  0.154435  0.673574  0.146318

'''
#  페어와이즈 삭제 (pairwise delection)
: 결손이 적은 열만 남기는 것
'''
print(sample_data_frame[[0, 1, 2]].dropna())
#           0         1         2
# 0  0.331314  0.510333  0.296017
# 3  0.573541  0.037328  0.077334
# 4  0.936057  0.501714  0.265234
# 5  0.139453  0.853888  0.589892
# 6  0.514939  0.481430  0.386608
# 7  0.612340  0.307514  0.771854
# 8  0.793148  0.822876  0.622429
# 9  0.945902  0.531969  0.306064


# 문제 
print(sample_data_frame[[0, 2]].dropna())   # 해당 열만 남기고 버림
#           0         2
# 0  0.480231  0.506938
# 3  0.195454  0.602821
# 4  0.584765  0.052796
# 5  0.895301  0.380775
# 6  0.402695  0.852949
# 7  0.738165  0.913220
# 8  0.538291  0.256736
# 9  0.855331  0.178076



##############################14.3.2 결측치 보완 ##############################

import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA


# NaN부분에 0을 채워줌
print(sample_data_frame.fillna(0))
#           0         1         2         3
# 0  0.335148  0.841119  0.807990  0.852771
# 1  0.000000  0.047669  0.054428  0.804650
# 2  0.546235  0.838710  0.000000  0.032273
# 3  0.873690  0.111776  0.782563  0.383145
# 4  0.301481  0.587382  0.068532  0.730293
# 5  0.232170  0.909259  0.590366  0.000000
# 6  0.356321  0.441416  0.787681  0.000000
# 7  0.882611  0.542004  0.123339  0.000000
# 8  0.274270  0.783789  0.637452  0.000000
# 9  0.405601  0.824665  0.145514  0.000000


# 앞의 값으로 채워줌
print(sample_data_frame.fillna(method='ffill'))
#           0         1         2         3
# 0  0.698015  0.849745  0.774609  0.871161
# 1  0.698015  0.343800  0.989875  0.428989
# 2  0.346494  0.828481  0.989875  0.887870
# 3  0.116606  0.432310  0.545918  0.770473
# 4  0.570169  0.537280  0.190131  0.648806
# 5  0.954537  0.488866  0.308933  0.648806
# 6  0.939849  0.932605  0.228400  0.648806
# 7  0.211909  0.473193  0.240804  0.648806
# 8  0.597562  0.518519  0.215010  0.648806
# 9  0.106549  0.905054  0.079002  0.648806


##############################14.3.3 결측치 보완(평균값 대입법) ##############################

import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA


# 평균값 대입법
print(sample_data_frame.fillna(sample_data_frame.mean()))
#           0         1         2         3
# 0  0.582127  0.658194  0.355504  0.139656
# 1  0.447608  0.850627  0.229963  0.116295
# 2  0.240036  0.795527  0.442877  0.711758
# 3  0.644237  0.456156  0.254392  0.034592
# 4  0.307060  0.061133  0.587649  0.177947
# 5  0.325356  0.014334  0.623631  0.236049
# 6  0.605752  0.909841  0.768462  0.236049
# 7  0.230069  0.283802  0.231814  0.236049
# 8  0.434539  0.221748  0.725991  0.236049
# 9  0.659295  0.155886  0.208487  0.236049


# 문제 
np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

print(sample_data_frame.fillna(sample_data_frame.mean()))
#           0         1         2         3
# 0  0.548814  0.715189  0.602763  0.544883
# 1  0.531970  0.645894  0.437587  0.891773
# 2  0.963663  0.383442  0.791725  0.528895
# 3  0.568045  0.925597  0.071036  0.087129
# 4  0.020218  0.832620  0.778157  0.870012
# 5  0.978618  0.799159  0.461479  0.780529
# 6  0.118274  0.639921  0.523791  0.944669
# 7  0.521848  0.414662  0.523791  0.774234
# 8  0.456150  0.568434  0.523791  0.617635
# 9  0.612096  0.616934  0.523791  0.681820

##############################14.4.1 키별 통계량 산출 ##############################

import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header =None)
# 각 수치가 무엇을 나타내는지 컬럼 헤더를 추가합니다.
df.columns =[ "", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Mangnesium", 
            "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", 
            "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]


# 평균값 구하기
print(df['Alcohol'].mean())      # 13.000617977528083

# 문제
print(df["Mangnesium"].mean())   # 99.74157303370787


##############################14.4.2 중복 데이터 ##############################


import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({'col1':[1, 1, 2, 3, 4, 4, 6, 6],
                        'col2': ['a','b','b','b','c','c','b','b']})

print(dupli_data)
#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 5     4    c
# 6     6    b
# 7     6    b


''' .duplicated() '''
# 중복된 행을 True로 표시
print(dupli_data.duplicated())
# 0    False
# 1    False
# 2    False
# 3    False
# 4    False
# 5     True
# 6    False
# 7     True
# dtype: bool

''' .drop_duplicates() '''
# 중복 데이터가 삭제된 후의 데이터를 보여줌
print(dupli_data.drop_duplicates())
#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 6     6    b

##############################14.4.3 매핑  ##############################

import pandas as pd
from pandas import DataFrame

attri_data1 = {'ID':['100','101','102','103','104','106','108','110','111','113'],
               "city": ["서울",'부산','대전','광주','서울','서울','부산','대전','광주','서울'],
               "brith_day":[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name":['영이','순돌','짱구','태양','션','유리','현아','태식','민수','호식']}

atrri_data_frame1 = DataFrame(attri_data1)

print(atrri_data_frame1)
#     ID city  brith_day name
# 0  100   서울       1990   영이
# 1  101   부산       1989   순돌
# 2  102   대전       1992   짱구
# 3  103   광주       1997   태양
# 4  104   서울       1982    션
# 5  106   서울       1991   유리
# 6  108   부산       1988   현아
# 7  110   대전       1990   태식
# 8  111   광주       1995   민수
# 9  113   서울       1981   호식


city_map = {'서울':'서울',
            '광주': '전라도',
            '부산': '경상도',
            '대전':'충청도'}

print(city_map)
# {'서울': '서울', '광주': '전라도', '부산': '경상도', '대전': '충청도'}


''' mapping '''
# 새로운 컬럼 region을 추가합니다. 해당 데이터가 없는 경우 NaN
atrri_data_frame1['region'] = atrri_data_frame1['city'].map(city_map)
print(atrri_data_frame1)
#     ID city  brith_day name region
# 0  100   서울       1990   영이     서울
# 1  101   부산       1989   순돌    경상도
# 2  102   대전       1992   짱구    충청도
# 3  103   광주       1997   태양    전라도
# 4  104   서울       1982    션     서울
# 5  106   서울       1991   유리     서울
# 6  108   부산       1988   현아    경상도
# 7  110   대전       1990   태식    충청도
# 8  111   광주       1995   민수    전라도
# 9  113   서울       1981   호식     서울


# 문제
MS_map = {'서울':'중부',
        '광주': '남부',
        '부산': '남부',
        '대전':'중부'}

atrri_data_frame1['MS'] = atrri_data_frame1['city'].map(MS_map)
print(atrri_data_frame1)
#     ID city  brith_day name region  MS
# 0  100   서울       1990   영이     서울  중부
# 1  101   부산       1989   순돌    경상도  남부
# 2  102   대전       1992   짱구    충청도  중부
# 3  103   광주       1997   태양    전라도  남부
# 4  104   서울       1982    션     서울  중부
# 5  106   서울       1991   유리     서울  중부
# 6  108   부산       1988   현아    경상도  남부
# 7  110   대전       1990   태식    충청도  중부
# 8  111   광주       1995   민수    전라도  남부
# 9  113   서울       1981   호식     서울  중부


##############################14.4.4 구간 분할  ##############################

import pandas as pd
from pandas import DataFrame

attri_data1 = {'ID':['100','101','102','103','104','106','108','110','111','113'],
               "city": ["서울",'부산','대전','광주','서울','서울','부산','대전','광주','서울'],
               "brith_year":[1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name":['영이','순돌','짱구','태양','션','유리','현아','태식','민수','호식']}

atrri_data_frame1 = DataFrame(attri_data1)


# 분할 리스트 생성
birth_year_bins = [1980, 1985, 1990, 1995, 2000]

''' pd.cut() '''
# 구간 분할 실시
birth_year_cut_data = pd.cut(atrri_data_frame1.brith_year, birth_year_bins)

print(birth_year_cut_data)
# 0    (1985, 1990]
# 1    (1985, 1990]
# 2    (1990, 1995]
# 3    (1995, 2000]
# 4    (1980, 1985]
# 5    (1990, 1995]
# 6    (1985, 1990]
# 7    (1985, 1990]
# 8    (1990, 1995]
# 9    (1980, 1985]
# Name: brith_year, dtype: category
# Categories (4, interval[int64]): [(1980, 1985] < (1985, 1990] < (1990, 1995] < (1995, 2000]]


''' .value_counts() '''
# 각 구간의 수 집계
print(pd.value_counts(birth_year_cut_data))
# (1985, 1990]    4
# (1990, 1995]    3
# (1980, 1985]    2
# (1995, 2000]    1
# Name: brith_year, dtype: int64


''' labels '''
group_names = ['first1980', 'secaond1980', 'first1990', 'second1990']
birth_year_cut_data = pd.cut(atrri_data_frame1.brith_year, birth_year_bins, labels = group_names)
print(pd.value_counts(birth_year_cut_data))
# secaond1980    4
# first1990      3
# first1980      2
# second1990     1
# Name: brith_year, dtype: int64


''' pd.cut( , n) : 분할수 지정 '''
print(pd.cut(atrri_data_frame1.brith_year, 2))
# 0      (1989.0, 1997.0]
# 1    (1980.984, 1989.0]
# 2      (1989.0, 1997.0]
# 3      (1989.0, 1997.0]
# 4    (1980.984, 1989.0]
# 5      (1989.0, 1997.0]
# 6    (1980.984, 1989.0]
# 7      (1989.0, 1997.0]
# 8      (1989.0, 1997.0]
# 9    (1980.984, 1989.0]
# Name: brith_year, dtype: category
# Categories (2, interval[float64]): [(1980.984, 1989.0] < (1989.0, 1997.0]

