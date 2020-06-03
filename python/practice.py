print('풍선')
print("나비")
print("ㅋ"*10)

#''''''''''''''''''''''1. 참 / 거짓''''''''''''''''''''''

print(5 > 10)
print(5 < 10)
print(True)
print(False)
print(not True)
print(not False)
print(not 5 > 10)


#''''''''''''''''''''''애완동물을 소개해 주세요'''''''''''''''''''''

animal = "고양이" 
name = "해피"
age = 4
hobby = "공놀이"
is_adult = age >= 3



print("우리집" +animal+"의 이름은 " +name+"예요")
print(name+"는 " + str(age) + "살이며," + hobby +"을 아주 좋아해요")
# print(name,"는 ",  str(age), "살이며,",hobby,"을 아주 좋아해요") #  ,를 쓰면 빈칸이 들어간다.

print(name+"는 어른일까요?" + str(is_adult))

#str = string(문자열)이란 문자, 단어 등으로 구성된 문자들의 집합
#" " 큰따옴표로 둘러싸여 있으면 모두 문자열이라고 보면 된다. 


'''Quiz) 변수를 이용하여 다음 문장을 출력하시오

변수명 : station

변수값 : "사당", "신도림", "인천공항" 순서대로 입력 

출력문장 : XX행 열차가 들어오고 있습니다.

'''

station = "인천공항"

print(station + "행 열차가 들어오고 있습니다" )


#''''''''''''''''''''''2.연산자''''''''''''''''''''''


print(1+1)           #2
print(3-2)           #1
print(5*2)           #10
print(6/3)           #2

print(2**3)          #2^3 = 8
print(5%3)           #나머지 구하기 2
print(10%3)          # 나머지 1
print(5//3)          #몫 구하기  =1
print(10//3)         #3
print(10>3)          #True
print(4 >= 7)        #False
print(10 < 3)        #False
print(5 <= 5)        #True

print(3 == 3 )       # 3은 3과 똑같다 True
print(4 == 3)        #False
print(3 + 4 == 7)    #True

print(1 != 3)        #1과 3은 같지 않다. True
print(not(1!=3))     #False

print((3 > 0) and (3 < 5))    # 모두 만족  True
print((3 > 0) & (3 < 5))    # 모두 만족  True

print((3 > 0) or (3 > 5))   # or 조건은 둘중하나만 참이면 참 
print((3 > 0) | (3 > 5))   # or 조건은 둘중하나만 참이면 참 


print(2 + 3 * 4 )    #14
print((2 + 3) *4)    #20

number = 2 + 3 * 4   #14
print(number)

number = number + 2  #16
print(number)

number += 2          #18
print(number)

number *= 2          #36
print(number)

number /= 2          #18
print(number)

number -= 2          #16
print(number)


number %= 2          #0
print(number)


#''''''''''''''''''''''3.절대값'''''''''''''''''''''


print(abs(-5))      #5
print(pow(4, 2))    #4^2 = 4*4 = 16
print(max(4, 12))   #12
print(min(4, 12))   #4
print(round(3.14))  #round는 반내림 or 반올림 = 3
print(round(4.99))  #5


#''''''''''''''''''''''4.math 라이브러리 사용''''''''''''''''''''''
from math import *
print(floor(4.99))  #내림 4
print(ceil(3.14))   #올림 4
print(sqrt(16))     #제곱근 4 


#''''''''''''''''''''''5.random 라이브러리 사용''''''''''''''''''''''

from random import *

print(random())      #랜덤함수를 통해서 난수를 뽑음 0.0~ 1.0 미만의 임의의 값 생성
print(random() * 10)       #0.0 ~ 1.0 미만의 임의의 값 생성
print(int(random() * 10))  #0.0 ~ 10 미만의 임의의 값 생성 
print(int(random() * 10))  #0.0 ~ 10 미만의 임의의 값 생성  
print(int(random() * 10))  #0.0 ~ 10 미만의 임의의 값 생성
print(int(random() * 10) + 1)  #1 ~ 10 이하의 임의의 값 생성
print(int(random() * 10) + 1)  #1 ~ 10 이하의 임의의 값 생성
print(int(random() * 10) + 1)  #1 ~ 10 이하의 임의의 값 생성

print(int(random() * 45) + 1)  #1 ~ 45 이하의 임의의 값 생성
print(int(random() * 45) + 1)  #1 ~ 45 이하의 임의의 값 생성
print(int(random() * 45) + 1)  #1 ~ 45 이하의 임의의 값 생성

print(randrange(1, 45))        #1 ~ 45 미만의 임의의 값 생성
print(randrange(1, 45))        #1 ~ 45 미만의 임의의 값 생성
print(randrange(1, 45))        #1 ~ 45 미만의 임의의 값 생성

print(randint(1, 45))              #1 ~ 45 이하의 임의의 값 생성
print(randint(1, 45))              #1 ~ 45 이하의 임의의 값 생성
print(randint(1, 45))              #1 ~ 45 이하의 임의의 값 생성


''' Quiz)당신은 최근에 코딩 스터디 모임을 새로 만들었습니다.
    월 4회 스터디를 하는데 3번은 온라인으로 하고 1번은 오프라인으로 하기로 했습니다.
    아래 조건에 맞는 오프라인 모임 날짜를 정해주는 프로그램을 작성하시오.

    조건1 : 랜덤으로 날짜를 뽑아야 함
    조건2 : 월별 날짜는 다름을 감안하여 최소 일수인 28일 이내로 정함
    조건3 : 매월 1~3일은 스터디 준비를 해야 하므로 제외 

    (출력문 예제)
    오프라인 스터디 모임 날짜는 매월 x일로 선정되었습니다.
'''

from random import *

date = randint(4, 28)
print("오프라인 스터디 모임 날짜는 매월 "+str(date)+"일로 선정되었습니다")


#''''''''''''''''''''''6.문자열''''''''''''''''''''''

sentence = '나는 소년입니다'
print(sentence)
sentence2 = "파이썬은 쉬워요"
print(sentence2)
sentence3 = """
나는 소년이고
파이썬은 쉬워요
"""
print(sentence3)

#''''''''''''''''''''''6.슬라이싱'''''''''''''''''''''

jumin = "890528-1149613"

print("성별: " + jumin[7])
print("연 : " + jumin[0:2])  # 0 부터 2 직전까지 
print("월 : " + jumin[2:4])  # 2 부터 4 직전까지
print("일 : " + jumin[4:6])  # 4 부터 6 직전까지
print("생년월일 : " +jumin[:6])  #  :6 = 처음부터 6직전까지

ㅇ



























