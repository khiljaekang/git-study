#print문과 format 함수

a = '사과'
b = '배'
c = '옥수수'

print('선생님은 잘생기셨다.')

print(a)
print(a, b)
print(a, b, c)

print("나는 {0}를 먹었다.".format(a)) # (") 큰 따옴표로 둘러싸인 문자열은 +연산과 동일하다.
print("나는 {0}와 {1}를 먹었다,".format(a, b))
print("나는 {0}와 {1}와 {2}를 먹었다".format(a, b, c))


print('나는',a,'를 먹었다.')
print('나는',a,'와',b,'를 먹었다')
print('나는',a,'와',b,'와',c,'를 먹었다')
