############파이썬#############

#2.4 들여쓰기 
for i in [1,2,3,4,5,]:
    print(i)

    for j in [1,2,3,4,5,]:
        print(j)
        print(i+j) 

#2.6 함수
def double(x):
    x = 2
    '''
    입력된 변수에 2를 곱한 값을 출력해 준다
    '''
    
def apply_to_one(f):
    '''
    인자가 1인 함수 f를 호출
    '''

'''
# 함수는 인자에 기본값을 할당할 수 있는데, 기본값 외의 값을 전달하고 싶을 때는 값을 직접 명시해 주면 된다.
'''

def my_print(message = "my default message"):
    print('hello')             #hello를 출력
    print()                    #'my default message'를 출력

#2.9 리스트

'''
대괄호를 사용해 리스트의 n번째 값을 불러오거나 설정할 수 있다.

x = [0,1,2,3,4,5,6,7,8,9]

zero = x[0]                    #결과는 0, 리스트의 순서는 0부터 시작한다.
one = x[2]                     #결과는 1
nine = x[-1]                   #결과는 9, 리스트의 마지막 항목을 불러온다.
eight = x[-2]                  #결과는8, 뒤에서 두번째의 항목을 불러온다.


x = [1,2,3]
x.extend([4,5,6])              #는 이제 [1,2,3,4,5,6]

x = [1,2,3] 
y = x + [4,5,6]                #이제 y는 [1,2,3,4,5,6] 이며 x는 변하지 않았다.

'''

#2.10 튜플

'''
이해못함
'''

#2.11 딕셔너리
'''
딕셔너리는 파이썬의 또 다른 기본적인 데이터 구조이며, 특정값과 연관된 키를 연결해 주고 이를 사용해
값을 빠르게 검색할 수 있다.

empty_dict = {}
empty_dict2 =()
grades = "Joel": 80, "Tim":95}

Joels_grade = grades["Joel"]            #결과는 80

Kates_grade = grades["Kate"]            #존재하지 않는 키를 입력하면 keyerror가 발생한다.

#연산자 in을 사용하면 키의 존재 여부를 확인할 수 있다.

joel_has_grade = "Joel" in grades       #True
kate has grade = "Kate" in grades       #False

'''

#2. 13 Set
'''
집합(set)은 파이썬의 데이터 구조 중 유일한 항목의 집합을 나타내는 구조다. 집합은 중괄호를 사용해서 정의한다.

s = set()
s.add(1)     #는 이제 { 1 }
s.add(2)     #는 이제 {1, 2}
x = len(s)   #결과는 2
y = 2 in s   #True
z = 3 in s   #False  

'''


#2. 14 흐름제어
'''
대부분의 프로그래밍 언어처럼 if를 사용하면 조건에 따라 코드를 제어할 수 있다. 

if 1 > 2:
    message = "if only 1 were greater than two..."
elif 1 > 3:
    message = "elif stands for 'else i"
else:
    message = "when all else fails use else (if you want to)"

파이썬에도 while이 존재하지만 

x = 0
while x < 10:
    print("f{x} is less than 10")
    x += 1

다음과 같이 for와 in을 더 자주 사용할 것이다.

for x range(10):
    print(f"{x} is less than 10")

만약 더 복잡한 논리 체계가 필요하다면 continue와 break를 사용할 수 있다.

for x in range(10):
    if x == 3:
        continue      #다음경우로 넘어간다.
    if x == 5:
        break         #for문 전체를 끝낸다.     
    print(x)          # 0,1,2,4
'''

#2.16 정렬

'''
파이썬의 모든 리스트에는 리스트를 자동으로 정렬해 주는 sort 메서드가 있다. 만약 이미 만든 리스트를 망치고
싶지 않다면, sorted 함수를 사용해서 새롭게 정렬된 리스트를 생성할 수 있다.

x=[4,1,2,3]
y= sorted(x)          #y는 [1,2,3,4]   하지만 x는 변하지 않는다.
x.sort()              #이제 x는 [1,2,3,4]

기본적으로 sort 메서드와 sorted 함수는 리스트의 각 항목을 일일이 비교해서 오름차순으로 정렬해 준다.
만약 리스트를 내림차순으로 정렬하고 싶다면 인자에 reverse=True 를 추가해서 오름차순으로 정렬해 준다.
그리고 리스트의 각 항목끼리 서로 비교하는 대신 key를 사용하면 지정한 함수의 결괏값을 기준으로 리스트를 
정렬할 수 있다.

#절댓값의 내림차순으로 리스트를 정렬
x = sorted ([-4,-3,-2,3]), key=abs, reverse=true)  #결과는 [-4,-3,-2,1]

'''

