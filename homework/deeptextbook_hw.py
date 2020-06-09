#함수 기초

##############################6.1내장함수와 메서드##############################

##############################6.1.1 내장 함수##################################

'''
(1)함수 : 처리를 정리한 프로그램 
(2)내장 함수 : 대표적인 예 = print(),type(),int(),str()    
(3)객체or오브젝트 : 변수에 할당할 수 있는 요수. 대입되는 값을 인수라 부르며, 인수를 파라미터라고 부르는 경우도 있다.
TIP.함수에서 인수를 받는 변수의 자료형은 정해져 있다.
    ex) len() 함수는 문자열형(str형), 리스트형(list형)은 인수로 받을 수 있지만, 
        정수형(int), 부동소수점형(float형), 불리언형(bool형) 등은 인수로 받을 수 없다. 
        인수를 확인할 때에는 파이썬의 레퍼런스를 참조하는 것이 좋다
        ## a = 1 , b = a 라고 할때, a가 변수이고, 1이 인수.
        
    *오류가 발생하지 않는 예 : len("tomato")  #6  ,
                             len([1,2,3,])  #3
    
    *오류가 발생하는 예 : len(3),  len(2.1), len(True)
                                             
'''
##############################6.1.2 메서드##################################
'''
(1)메서드 : 어떠한 값에 대해 처리를 뜻하는 것이며, '값.메서드명()'형식으로 기술한다. 역할은 함수와 동일
           함수의 경우는 처리하는 값을 () 안에 기입했지만, 메서드형은 값 뒤에 .(점)을 연결
           함수와 마찬가지로 값의 자료형에 따라 사용할 수 있는 메서드가 다름.
           append () 는 리스트형에 사용할 수 있는 메서드
           
           TIP.함수와 메서드의 예(1)
               ##sorted함수 : 정렬함수
               number = [1,3,5,2,4]
               print(sorted(number))      # [1,3,5,2,4]
               print(number)              # [1,2,3,4,5]     
               
               number = [1,3,5,2,5]
               number.sort()
               print(number)              # [1,2,3,4,5]
                  
''' 
##############################6.1.3 문자열형 메서드(upper.count)##############################
'''
(1)upper : 모든 문자열을 대문자로 반환하는 메서드 
(2)count : ()안에 들어 있는 문자열에 요소가 몇 개 포함되어 있는지 알려주는 메서드
           TIP. ex)city = "Tokyo"
                print(city.upper())        #TOKYO
                print(city.count("o"))     #2 
'''
##############################6.1.4 문자열형 메서드(format)##############################
'''
(1)format : format() 메서드는 임의의 값을 대입한 문자열을 생성할 수 있다. 
            문자열 내에 {} 를 포함하는 것이 특징.
             
            Tip. ex) print("나는 {}에서 태어나 {}에서 유년기를 보냈습니다.").format("인천", "관교동"))
            
'''            
##############################6.1.5 리스트형 메서드(index)##############################
'''       
(1)index : 리스트형에는 인덱스 번호가 존재한다. 인덱스 번호는 리스트 내용을 0부터 순서대로 나열했을 때의 번호.
           Tip. ex) alphabet = ["a","b","c","d","d"]
                print(alphabet.index("a"))               #0
                print(alphabet.count("d"))               #2           
'''
##############################6.4 문자열 포맷 지정##############################
'''
(1)%d : 정수로 표시
(2)%f : 소수로 표시
(3).2f: 소수점 이하 두 자리까지 표시
(4)$s : 문자열로 표시 
'''
##############################7. NumPy 개요##############################


##############################7.1.1 Numpy가 하는 일##############################
'''
(1)NumPy : numpy는 파이썬으로 벡터나 행렬 계산을 빠르게 하도록 특화된 기본 라이브러리
(2)라이브러리 : 외부에서 읽어 들이는 파이썬 코드 묶음  ex) pandas, skcit-learn, matploplib 등 
'''
##############################7.2.3 1차원 배열의 계산##############################
'''
Tip. 1차원 배열 계산의 예 
###numpy를 사용하지 않고 실행###
storages = [1, 2, 3, 4]
new_storages = []
for n in storage:
    n += n
    new_storages.append(n)
print(new_storages)             # [2, 4, 6, 8]   
###numpy를 사용하여 실행###
import numpy as np
storages = np.array[1, 2, 3, 4]
storages += storages
print(storages)                 # [2, 4, 6, 8]
'''

##############################7.2.4 인덱스 참조와 슬라이스##############################
'''
Tip. 슬라이스의 예 
arr = np.arange(10)
print(arr)                    #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
arr = np.arange(10)
arr[0:3] = 1
print(arr)                    #[0, 1, 1, 1, 4, 5, 6, 7, 8, 9 ]
import numpy as np
arr = np.arange(10)
print(arr[3:6])               #[3, 4, 5]
arr[3:6] = 24
print(arr)                    #[0, 1, 2, 3, 24, 24, 24, 6, 7, 8, 9]
'''
##############################7.2.8 범용 함수##############################
'''
(*)범용함수 : ndarray 배열의 각 요소에 대한 연산 결과를 반환하는 함수
1.범용하는 인수가 하나인 경우
(1)np.abs() : 절대값을 반환
(2)np.exp() : 요소의 거듭제곱을 반환
(3)np.sqrt(): 요소의 제곱근을 반환
2.범용하는 인수가 두개인 경우
(1)np.add() : 요소간의 합을 반환
(2)np.subtract() : 요소간의 차를 반환
(3)np.maximum() : 요소간의 최대값을 반환
'''
##############################7.2.9 집합 함수##############################
'''
(1)np.unique() : 배열 요소에서 중복을 제거하고, 정렬한 결과를 반환
(2)np.union1d(x,y): 배열 x와, y의 합집합을 정렬해서 반환
(3)np.intersect1d(x,y) : 배열x와 y의 교집합을 정렬해서 반환
(4)np.setdiff1d(x,y) : 배열 x에서 y를 뺀 차집합을 정렬해서 반환 
'''
##############################7.2.10 난수##############################

'''
(1)np.random.rand() : 0이상 1미만의 난수를 생성
(2)np.random.randint(x, y, z) : x이상 y미만의 정수를 z개 생성
(3)np.random.normal() : 가우스 분포를 따르는 난수 생성
'''
##############################7.3.1 NumPy 2차원 배열##############################

'''
* 2차원 배열은 행렬에 해당한다.
ex) arr = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])             
    print(arr)                      #[[1, 2, 3, 4]
                                      [5, 6, 7, 8]]
    
    print(arr.shape)                #(2,4)
    
    print(arr.reshape(4, 2))                                    
'''
##############################7.3.2 인덱스 참조와 슬라이스##############################
'''
arr = np.array([[1, 2, 3],[4, 5, 6]])
print(arr[1])                            # [4, 5, 6]
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1,2])                          # 6
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1,1:])                         # [5, 6]                  
'''
##############################7.3.3 axis ##############################
'''
* axis : 좌표축과 같다. 행마다 처리하는 축이 axis = 1, 열마다 처리하는 축이 axis = 0
'''


##############################7.3.5 전치 행렬 ##############################
'''
*전치 : 행렬에서 행과 열을 바꾸는 것. 
        ex) np.transpose(), .T
            
'''
##############################7.3.6 정렬 ##############################
'''
ndarray도 리스트형과 마찬가지로 sort() 메서드로 정렬할 수 있음.
2차원 배열의 경우 0을 인수로 하면 열단위로 요소가 정렬되며, 
1을 인수로 하면 행단위로 요소로 정렬
np.sort = 정렬 된 배열의 복사본을 반환
argsort() = 정렬 된 배열의 인덱스 반환             #머신 러닝에서 자주 사용
arr = np.array([15, 30, 5])
arr.argsort()                        # arr([2, 0, 1])
arr = np.array([[8, 4, 2], [3, 5, 1]])
print(arr.argsort())                # [2, 1, 0], [2, 0, 1]
print(np.sort(arr))                 # [2, 4, 8], [1, 3, 5]
arr.sort(1)
print(arr)                          # [2, 4, 8], [1, 3, 5]
'''
##############################7.3.8 통계 함수 ##############################
'''
(*)통계 함수 : ndarray 배열 전체 또는 특정 축을 중심으로 수학적 처리를 수행하는 함수 또는 메서드
(1):mean() : 배열 요소의 평균반환
(2):np.average() : 배열 요소의 평균반환
(3)np.max() : 최대값
(4)np.min() : 최소값
(5)np.argmax() : 요소의 최대값의 인덱스 번호 반환
(6)np.argmin() : 요소의 최소값의 인덱스 번호 반환 
        
'''
##############################8.1.Pandas 개요 ##############################
'''
(1)pandas : pandas 는 일반 적인 데이터 베이스에서 이뤄지는 작업을 수행할 수 있으며,
            수치뿐 아니라 이름과 주소등 문자열 데이터도 쉽게 처리할 수 있다.
            pandas 는 series와 dataframe의 두가지 데이터의 구조가 존재하며,
            행의 라벨은 인덱스, 열의 라벨은 컬럼이라고 한다. 
'''
##############################8.1.Series와  DataFrame의 데이터 확인 ##############################
'''
(1)series 예 : import pandas as pd
               fruit = {"orange" :2, "banana" : 3}      
               print(pd.Series(fruits))                     # banana 3 , orange 2
                            
       
(2)dataframe 예 : import pandas as pd
                  data = {"fruit": ["apple", "orange", "banana", strawberry", "kiwi fruit"]      #        fruits  time  year
                          "year" : [2001, 2002, 2001, 2008, 2006],                                   0     apple    1   2001
                          "time" : [1, 4, 5, 6, 3]}                                                  1    orange    4   2002
                  df = pd.DataFrame(data)                                                            2    banana    5   2001
                  print(df)                                                                          3  strawberry  6   2008
                                                                                                     4  kiwifruits  3   2006
                                                                                                     
                                                                                                     
(3)-인덱스 참조를 사용하여 series의 2~4번째에 있는 세 요소를 추출하여 items1에 대입하시오
   -인덱스 값을 지정하는 방법으로 "apple","banana","kiwifruits"의 인덱스를 가진 요소를 추출하여 items2에 대입하시오.
   
import pandas as pd 
index = ["apple", "orange", "banana", strawberry", "kiwi fruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)
items1 = series[1:4]
items2 = series[["apple", "banana", "kiwi fruit"]]                                                                                                                                                                  
'''
##############################8.2.4. 요소 추가 ##############################
'''
Tip. 요소를추가하는 예
fruit = {"banana" : 3, "orange" : 2}
series = pd.Series(fruits)
series = series.append(pd.Series([3], index=["grape"]))
'''

##############################8.2.5 요소 삭제 ##############################
'''
(1)ex) series.drop("strawberry)    : drop.("인덱스")하여 인덱스 위치의 요소를 삭제할 수 있다.
'''
##############################8.2.6 필터링 ##############################
'''
(*)pandas에서는 bool형의 시퀀스를 지정해서 True인 것만 추출할 수 있다.
index = ["apple", "orange", "banana", strawberry", "kiwi fruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)
conditions = [True, true, False, False, False]
print(series(conditions))
'''
##############################8.2.7 정렬 ##############################
'''
(1)series.sort_index() : 인덱스 정렬
(2)series.sort_values(): 데이터 정렬
(3)ascending = False : 내림차순 정렬       #특별히 지정하지 않으면 오름차순으로 정렬 . accending의 디폴트 값은 True
'''
##############################8.3 DataFrame 생성 ##############################
'''
data = {"fruits" = ["apple", "orange", "banana", strawberry", "kiwi fruit"],
        "year"   = [2001, 2002, 2001, 2008, 2006]
        "time"   = [1, 4, 5, 6, 3]
df = pd.DataFrame(data)
print(df)
'''
##############################8.3.4 열추가 ##############################
'''
data = {"fruits" = ["apple", "orange", "banana", strawberry", "kiwi fruit"],
        "year"   = [2001, 2002, 2001, 2008, 2006]
        "time"   = [1, 4, 5, 6, 3]
df = pd.DataFrame(data)
df["price"] = [150, 120, 100, 300, 150]
print(df)
'''
##############################8.3.6 행 또는 열삭제 ##############################
'''
*df.drop() : 인덱스 또는 컬럼을 지정하여 해당 행 또는 열을 삭제
             열을 삭제하려면 두번제 인수로 axis = 1을 전달 해야함.
'''

#함수 기초
##############################6.1내장함수와 메서드##############################

##############################6.1.1 내장 함수##################################

'''
(1)함수 : 처리를 정리한 프로그램 
(2)내장 함수 : 대표적인 예 = print(),type(),int(),str()    
(3)객체or오브젝트 : 변수에 할당할 수 있는 요수. 대입되는 값을 인수라 부르며, 인수를 파라미터라고 부르는 경우도 있다.

TIP.함수에서 인수를 받는 변수의 자료형은 정해져 있다.
    ex) len() 함수는 문자열형(str형), 리스트형(list형)은 인수로 받을 수 있지만, 
        정수형(int), 부동소수점형(float형), 불리언형(bool형) 등은 인수로 받을 수 없다. 
        인수를 확인할 때에는 파이썬의 레퍼런스를 참조하는 것이 좋다
        ## a = 1 , b = a 라고 할때, a가 변수이고, 1이 인수.
        
    *오류가 발생하지 않는 예 : len("tomato")  #6  ,
                             len([1,2,3,])  #3
    
    *오류가 발생하는 예 : len(3),  len(2.1), len(True)
                                             
'''

##############################6.1.2 메서드##################################
'''
(1)메서드 : 어떠한 값에 대해 처리를 뜻하는 것이며, '값.메서드명()'형식으로 기술한다. 역할은 함수와 동일
           함수의 경우는 처리하는 값을 () 안에 기입했지만, 메서드형은 값 뒤에 .(점)을 연결
           함수와 마찬가지로 값의 자료형에 따라 사용할 수 있는 메서드가 다름.
           append () 는 리스트형에 사용할 수 있는 메서드
           
           TIP.함수와 메서드의 예(1)
               ##sorted함수 : 정렬함수
               number = [1,3,5,2,4]
               print(sorted(number))      # [1,3,5,2,4]
               print(number)              # [1,2,3,4,5]     
               
               number = [1,3,5,2,5]
               number.sort()
               print(number)              # [1,2,3,4,5]
                  
'''
##############################6.1.3 문자열형 메서드(upper.count)##############################
'''
(1)upper : 모든 문자열을 대문자로 반환하는 메서드 
(2)count : ()안에 들어 있는 문자열에 요소가 몇 개 포함되어 있는지 알려주는 메서드

           TIP. ex)city = "Tokyo"
                print(city.upper())        #TOKYO
                print(city.count("o"))     #2 

'''
##############################6.1.4 문자열형 메서드(format)##############################
'''
(1)format : format() 메서드는 임의의 값을 대입한 문자열을 생성할 수 있다. 
            문자열 내에 {} 를 포함하는 것이 특징.
             
            Tip. ex) print("나는 {}에서 태어나 {}에서 유년기를 보냈습니다.").format("인천", "관교동"))
            
'''            
##############################6.1.5 리스트형 메서드(index)##############################
'''       
(1)index : 리스트형에는 인덱스 번호가 존재한다. 인덱스 번호는 리스트 내용을 0부터 순서대로 나열했을 때의 번호.

           Tip. ex) alphabet = ["a","b","c","d","d"]
                print(alphabet.index("a"))               #0
                print(alphabet.count("d"))               #2           
'''
##############################6.4 문자열 포맷 지정##############################
'''
(1)%d : 정수로 표시
(2)%f : 소수로 표시
(3).2f: 소수점 이하 두 자리까지 표시
(4)$s : 문자열로 표시 
'''
##############################7. NumPy 개요##############################


##############################7.1.1 Numpy가 하는 일##############################
'''
(1)NumPy : numpy는 파이썬으로 벡터나 행렬 계산을 빠르게 하도록 특화된 기본 라이브러리
(2)라이브러리 : 외부에서 읽어 들이는 파이썬 코드 묶음  ex) pandas, skcit-learn, matploplib 등 
'''
##############################7.2.3 1차원 배열의 계산##############################
'''
Tip. 1차원 배열 계산의 예 

###numpy를 사용하지 않고 실행###

storages = [1, 2, 3, 4]
new_storages = []
for n in storage:
    n += n
    new_storages.append(n)
print(new_storages)             # [2, 4, 6, 8]   


###numpy를 사용하여 실행###

import numpy as np
storages = np.array[1, 2, 3, 4]
storages += storages
print(storages)                 # [2, 4, 6, 8]
'''
##############################7.2.4 인덱스 참조와 슬라이스##############################
'''
Tip. 슬라이스의 예 

arr = np.arange(10)
print(arr)                    #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

arr = np.arange(10)
arr[0:3] = 1
print(arr)                    #[0, 1, 1, 1, 4, 5, 6, 7, 8, 9 ]

import numpy as np

arr = np.arange(10)
print(arr[3:6])               #[3, 4, 5]
arr[3:6] = 24
print(arr)                    #[0, 1, 2, 3, 24, 24, 24, 6, 7, 8, 9]
'''
##############################7.2.8 범용 함수##############################
'''
(*)범용함수 : ndarray 배열의 각 요소에 대한 연산 결과를 반환하는 함수

1.범용하는 인수가 하나인 경우
(1)np.abs() : 절대값을 반환
(2)np.exp() : 요소의 거듭제곱을 반환
(3)np.sqrt(): 요소의 제곱근을 반환

2.범용하는 인수가 두개인 경우
(1)np.add() : 요소간의 합을 반환
(2)np.subtract() : 요소간의 차를 반환
(3)np.maximum() : 요소간의 최대값을 반환
'''
##############################7.2.9 집합 함수##############################
'''
(1)np.unique() : 배열 요소에서 중복을 제거하고, 정렬한 결과를 반환
(2)np.union1d(x,y): 배열 x와, y의 합집합을 정렬해서 반환
(3)np.intersect1d(x,y) : 배열x와 y의 교집합을 정렬해서 반환
(4)np.setdiff1d(x,y) : 배열 x에서 y를 뺀 차집합을 정렬해서 반환 
'''
#############################7.2.10 난수##############################
'''
(1)np.random.rand() : 0이상 1미만의 난수를 생성
(2)np.random.randint(x, y, z) : x이상 y미만의 정수를 z개 생성
(3)np.random.normal() : 가우스 분포를 따르는 난수 생성
'''
##############################7.3.1 NumPy 2차원 배열##############################
'''
* 2차원 배열은 행렬에 해당한다.
ex) arr = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])             
    print(arr)                      #[[1, 2, 3, 4]
                                      [5, 6, 7, 8]]
    
    print(arr.shape)                #(2,4)
    
    print(arr.reshape(4, 2))                                    

'''
##############################7.3.2 인덱스 참조와 슬라이스##############################
'''
arr = np.array([[1, 2, 3],[4, 5, 6]])
print(arr[1])                            # [4, 5, 6]

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1,2])                          # 6

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1,1:])                         # [5, 6]                  
'''
##############################7.3.3 axis ##############################
'''
* axis : 좌표축과 같다. 행마다 처리하는 축이 axis = 1, 열마다 처리하는 축이 axis = 0
'''
##############################7.3.5 전치 행렬 ##############################
'''
*전치 : 행렬에서 행과 열을 바꾸는 것. 
        ex) np.transpose(), .T
            
'''
##############################7.3.6 정렬 ##############################
'''
ndarray도 리스트형과 마찬가지로 sort() 메서드로 정렬할 수 있음.
2차원 배열의 경우 0을 인수로 하면 열단위로 요소가 정렬되며, 
1을 인수로 하면 행단위로 요소로 정렬

np.sort = 정렬 된 배열의 복사본을 반환
argsort() = 정렬 된 배열의 인덱스 반환             #머신 러닝에서 자주 사용

arr = np.array([15, 30, 5])
arr.argsort()                        # arr([2, 0, 1])

arr = np.array([[8, 4, 2], [3, 5, 1]])

print(arr.argsort())                # [2, 1, 0], [2, 0, 1]

print(np.sort(arr))                 # [2, 4, 8], [1, 3, 5]

arr.sort(1)
print(arr)                          # [2, 4, 8], [1, 3, 5]
'''
##############################7.3.8 통계 함수 ##############################
'''
(*)통계 함수 : ndarray 배열 전체 또는 특정 축을 중심으로 수학적 처리를 수행하는 함수 또는 메서드

(1):mean() : 배열 요소의 평균반환
(2):np.average() : 배열 요소의 평균반환
(3)np.max() : 최대값
(4)np.min() : 최소값
(5)np.argmax() : 요소의 최대값의 인덱스 번호 반환
(6)np.argmin() : 요소의 최소값의 인덱스 번호 반환 
        
'''
##############################8.1.Pandas 개요 ##############################
'''
(1)pandas : pandas 는 일반 적인 데이터 베이스에서 이뤄지는 작업을 수행할 수 있으며,
            수치뿐 아니라 이름과 주소등 문자열 데이터도 쉽게 처리할 수 있다.
            pandas 는 series와 dataframe의 두가지 데이터의 구조가 존재하며,
            행의 라벨은 인덱스, 열의 라벨은 컬럼이라고 한다. 
'''
##############################8.1.Series와  DataFrame의 데이터 확인 ##############################
'''
(1)series 예 : import pandas as pd
               fruit = {"orange" :2, "banana" : 3}      
               print(pd.Series(fruits))                     # banana 3 , orange 2
                            
       
(2)dataframe 예 : import pandas as pd
                  data = {"fruit": ["apple", "orange", "banana", strawberry", "kiwi fruit"]      #        fruits  time  year
                          "year" : [2001, 2002, 2001, 2008, 2006],                                   0     apple    1   2001
                          "time" : [1, 4, 5, 6, 3]}                                                  1    orange    4   2002
                  df = pd.DataFrame(data)                                                            2    banana    5   2001
                  print(df)                                                                          3  strawberry  6   2008
                                                                                                     4  kiwifruits  3   2006
                                                                                                     
                                                                                                     
(3)-인덱스 참조를 사용하여 series의 2~4번째에 있는 세 요소를 추출하여 items1에 대입하시오
   -인덱스 값을 지정하는 방법으로 "apple","banana","kiwifruits"의 인덱스를 가진 요소를 추출하여 items2에 대입하시오.
   
import pandas as pd 
index = ["apple", "orange", "banana", strawberry", "kiwi fruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

items1 = series[1:4]
items2 = series[["apple", "banana", "kiwi fruit"]]                                                                
                                                                                                     
'''
##############################8.2.4. 요소 추가 ##############################
'''
Tip. 요소를추가하는 예
fruit = {"banana" : 3, "orange" : 2}
series = pd.Series(fruits)
series = series.append(pd.Series([3], index=["grape"]))
'''
##############################8.2.5 요소 삭제 ##############################
'''
(1)ex) series.drop("strawberry)    : drop.("인덱스")하여 인덱스 위치의 요소를 삭제할 수 있다.
'''

##############################8.2.6 필터링 ##############################
'''
(*)pandas에서는 bool형의 시퀀스를 지정해서 True인 것만 추출할 수 있다.

index = ["apple", "orange", "banana", strawberry", "kiwi fruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

conditions = [True, true, False, False, False]
print(series(conditions))
'''

##############################8.2.7 정렬 ##############################
'''
(1)series.sort_index() : 인덱스 정렬
(2)series.sort_values(): 데이터 정렬
(3)ascending = False : 내림차순 정렬       #특별히 지정하지 않으면 오름차순으로 정렬 . accending의 디폴트 값은 True
'''
##############################8.3 DataFrame 생성 ##############################
'''
data = {"fruits" = ["apple", "orange", "banana", strawberry", "kiwi fruit"],
        "year"   = [2001, 2002, 2001, 2008, 2006]
        "time"   = [1, 4, 5, 6, 3]
df = pd.DataFrame(data)
print(df)
'''
##############################8.3.4 열추가 ##############################
'''
data = {"fruits" = ["apple", "orange", "banana", strawberry", "kiwi fruit"],
        "year"   = [2001, 2002, 2001, 2008, 2006]
        "time"   = [1, 4, 5, 6, 3]
df = pd.DataFrame(data)
df["price"] = [150, 120, 100, 300, 150]
print(df)
'''
##############################8.3.6 행 또는 열삭제 ##############################
'''
*df.drop() : 인덱스 또는 컬럼을 지정하여 해당 행 또는 열을 삭제
             열을 삭제하려면 두번제 인수로 axis = 1을 전달 해야함.
'''