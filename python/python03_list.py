#자료형
#1. 리스트
#리스트는 여러가지 자료형으로 묶을 수 있다.
#numpy 는 딱 한가지 자료형을 사용해야 한다.
a = [1,2,3,4,5,]
b = [1,2,3,'a','b']

print(b)                         #[1,2,3,'a','b']

print(a[0] + a[3])               # 5
# print(b[0] + b[3])             #자료형이 달라서 오류
print(type(a))                   # list
print(a[-2])                     # 4
print(a[1:3])                    # [2, 3]

a = [1, 2, 3, ['a', 'b', 'c']]   
print(a[1])                      # 2
print(a[-1])                     # ['a','b','c']
print(a[-1][1])                  # b

#1-2. 리스트 슬라이싱

a = [1,2,3,4,5]
print(a[:2])                     # [1, 2] 처음부터 2까지 나온다 

#1-3. 리스트 더하기
a = [1, 2, 3]
b = [4, 5, 6]

print(a + b)                     # [ 1,2,3,4,5,6]

c = [7,8,9,10]

print(a + c)                     # [1,2,3,7,8,9,10]
print(a * 3)                     # [1,2,3,1,2,3,1,2,3]

# print(a[2] +'hi')              #타입에러 

print(str(a[2]) + 'hi')

f = '5'
# print(a[2] + f)                #타입에러

print((a[2]) + int(f))           # 8


#리스트 관련 함수
a = [1, 2, 3]                
a.append(4)                      #.append = 추가하다
print(a)                         # [1,2,3,4]

# a = a.append(5)                None   .append 할 때는 자기꺼에다 쓰면 안됨.

a = [1, 3, 4, 2]
a.sort()                         # sort = 정렬하다
print(a)                         # [1,2,3,4]

a.reverse()                      #reverse = 반대로하다
print(a)                         # [4,3,2,1]

print(a.index(3))               #index = 색인  ==a[3]
print(a.index(1))                              ==a[1]

a.insert(0, 7)                  #insert = 삽입하다  [7,4,3,2,1]
print(a)
a.insert(3, 3)
print(a)                        #[7,4,3,3,2,1]

a.remove(7)                     #remove = 삭제하다  [4,3,3,2,1] 
print(a)

a.remove(3)                     #3이 두개 이상일 경우 앞에 것이 지워짐  [4,3,2,1]
print(a)

#  a= 하고 함수명을 하면 전부다 에러가 뜬다. a.으로 간다
#a.append는 절대 까먹으면 안된다. 
#파이썬 슬라이싱, 인덱스, 어팬드 , 리스트 중요하다



