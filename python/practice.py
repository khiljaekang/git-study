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
print("뒤 7자리 : " +jumin[7:14]) # 7 부터 14직전까지 또는 7:  7부터 끝까지
print("뒤 7자리 (뒤에서 부터) : " + jumin[-7:]) #뒤에서 7번째부터 끝까지 



#''''''''''''''''''''''7.문자열 처리 함수'''''''''''''''''''''

python = "python is Amazing"
print(python.lower())
print(python.upper())
print(python[0].islower())
print(python[0].isupper())
print(len(python))
print(python.replace("python","Java"))

index = python.index("n")
print(index)
index = python.index("n", index + 1)
print(index)

print(python.find("java"))  # 값이 포함되어 있지 않을 때는 -1을 반환해준다.
# print(python.index("java"))  #index에서는 원하는 값이 없을 때는 오류가 난다.

print(python.count("n"))


#''''''''''''''''''''''8.문자열 포맷'''''''''''''''''''''

# print("a"+"b")
# print("a","b")

##################문자열 포맷 방법1####################

print("나는 %d살입니다." %20)  #d는 정수값만 집어넣을 수 있다.
print("나는 %s을 좋아해요"%"파이썬") #%s 는 string(문자형)을 의미한다 
print("apple은 %c로 시작해요" %"A")  #%c 는 한글자만 받겠다.

############ %s 는 만능이다. 숫자나 문자나 둘다 불러올 수 있음
print("나는 %s살입니다." %"20")
print("나는 %s색과 %s색을 좋아해요." % ("파란", "빨간"))



##################문자열 포맷 방법2####################

print("나는 {}살입니다.".format(20))
print("나는 {}색과 {}을 좋아해요.".format("파란","빨간"))
print("나는 {0}색과 {1}을 좋아해요.".format("파란","빨간"))



##################문자열 포맷 방법3####################

print("나는 {age}살이며, {color}색을 좋아해요".format(age =20, color="빨간"))
print("나는 {color}살이며, {age}색을 좋아해요".format(age =20, color="빨간"))


##################문자열 포맷 방법4####################

age = 20
color = "빨간"
print(f"나는 {age}살이며, {color}색을 좋아해요")



#''''''''''''''''''''''9.탈출 문자'''''''''''''''''''''

#\n : 줄바꿈
print("백문이 불여일견\n백견이 불여일타")

#저는 "나도코딩"입니다.
print("저는 '나도코딩'입니다")
print('저는 "나도코딩"입니다')
print("저는 \"나도코딩\"입니다")
print('저는 \'나도코딩\'입니다')

# \\ : 문장 내에서 \

# \r : 커서를 맨 앞으로 이동
print("Red Apple\rPine")

# \b : 한글자 삭제 

print("Redd\bApple")

# \t : tab 역할 

print("Red\tApple")


''' Quiz) 사이트별로 비밀번호를 만들어 주는 프로그램을 작성하시오
    예) http://naver.com
    규칙1 : http:// 부분은 제외 => naver.com
    규칙2 : 처음 만나는 점(.) 이후 부분은 제외 => naver
    규칙3 : 남은 글자 중 처음 세자리 + 글자 갯수(5) + 글자 내 'e'갯수(1) + "!"로 구성
    
    예)생성된 비밀번호 : nav51!
'''


url = "http://naver.com"

my_str = url.replace("http://","")       #규칙1

print(my_str)

my_str = my_str[:my_str.index(".")]      #규칙2  my_str[0:5] -> 0 ~ 5 직전까지. (0 ,1, 2, 3, 4,)
print(my_str)

password = my_str[:3] + str(len(my_str)) + str(my_str.count("e")) + "!"
print*"{0}의 비밀번호는 {1} 입니다.".format(url, password)









































