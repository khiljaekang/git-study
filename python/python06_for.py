#반복문
a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}

for i in a.keys():
    print(i)               #콜론을 치면 for문의 나와바리가 생긴다.



a = [1,2,3,4,5,6,7,8,9,10]
for i in a:
    i = i*i
    print(i)
    #print('kang')

for i in a:
    i = 1*1
    print(i)

##while문
'''
while 조건문 :     #참일 동안 계속 돈다.
    수행할 문장
'''

### if문                     


if 1 :
    print('True')
else :
    print('False')

if 3 :
    print('True')
else :
    print('False')

if 0 :
    print('True')
else :
    print('False')

if -1 :
    print('True')
else :
    print('False')

'''
비교연산자

< , >, ==, !=, >=, <=                     # == 은 비교할때 같다라는의미, != 은 같지않냐 다르냐 

'''

# if a = 1:                   
#     print(출력잘되)
a = 1
if a == 1:
    print('출력잘되')

money = 10000
if money >= 30000:
    print('우리한우먹자')
else:
    print('라면먹자')

### 조건연산자
# and, or, not              and는 둘다, or 둘중 하나, not은 둘다 아닌거

money = 20000
card = 1
if money >= 30000 or card == 1:
    print('한우먹자') 
else:
    print('라면먹자')


jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i >= 60:
        print("경]합격[축")
        number = number + 1
print("합격인원 : ", number, "명")

#########################################################################
# break, continue 
print("=====================break====================")
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 30:
        break
    if i >= 60:
        print("경]합격[축")
        number = number + 1
print("합격인원 : ", number, "명")

print("=====================continue====================") #조건에 맞는것만 그 다음으로 넘어간다.
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 60:
        continue
    if i >= 60:
        print("경]합격[축")
        number = number + 1
print("합격인원 : ", number, "명")








