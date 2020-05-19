#정수형

a = 1
b = 2
c = a + b
print(c)
d = a * b
print(d)

e = a / b
print(e) #파이썬은 정수와 정수를 나눴을 때, 실수가 나올 수 있다.

#실수형
a = 1.1
b = 2.2
c = a + b   #3.3000000000000003

print(c)

d = a*b     #2.4200000000000004

print(d)  

e = a/b  #0.5 

print(e) 

#3.3000000000000003 과 #2.4200000000000004 가 나오는 이유는 컴퓨터는 실수를 연산할 때 오류가 발생하기 때문
#실수는 유한개의 비트를 사용하여 근삿값으로 표현한다. 즉, 0.30000000000000004는 0.3을 표현한 근삿값이다.
#만약 두 실수가 같은지 판단할 때는 ==을 사용하면 안 된다.  다음과 같이 0.1 + 0.2와 0.3은 같지 않다고 나온다.

#문자형
a = 'hel'
b = 'lo'
c = a + b
print(c)

#문자 + 숫자      #타입에러        str = 문자형      int = 정수형  
a = 123
b = '45'
# c = a + b
# print(c)

#숫자를 문자변환 + 문자
a = 123
a = str(a)

print(a)

b = '45'
c = a + b

print(c) 

#문자를 숫자변환 + 문자  , 문자를 숫자로 숫자를 문자로 바꾸는 것을 형변환이라 한다.

a = 123
b = '45'
b = int(b)
c = a + b

print(c)

#문자열 연산하기 
#인덱스의 첫번째는 항상 0이다.
#시작은 0이고 끝은 뒤에서 부터 -1이다. 
a = 'abcdefgh'

print(a[0])
print(a[3])
print(a[5])
print(a[-1])
print(a[-2])
print(type(a)) #현재 타입을 알려줌.

b = 'xyz'
print(a + b)

# 문자열 인덱싱 
#띄어쓰기, 콤마, 마침표 등도 문자다.      # : , - 차이점에 주의해야 함.
a = 'Hello, Deep learning'
print(a[7])                     # D
print(a[-1])                    # g
print(a[-2])                    # n
print(a[3:9])                   # lo, De
print(a[3:-5])                  # lo, Deep lea
print(a[:-1])                   # Hello, Deep learnin
print(a[1:])                    # ello, Deep learning
print(a[5:-4])                  # , Deep lear








