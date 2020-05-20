#range함수
a = range(10)     
print(a)
b= range(1, 11)   
print(b)


for i in a:
    print(i)                      #0부터 9까지
for i in b:
    print(i)                      #1부터 10까지

print(type(a))

sum = 0
for i in range(1, 11):
    sum = sum + i
print(sum)