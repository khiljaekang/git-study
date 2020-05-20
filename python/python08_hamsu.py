print("==================덧셈==================")
def sum1(a, b):
    return a+b

a = 1
b = 2
c = sum1(a, b)

print(c)

### 곱셈, 나눗셈,뺄셈 함수를 만드시오.
###mull, div1, sub1
print("==================곱셈==================")
def mull1(a, b):
    return a*b

a = 1
b = 2
c = mull1(a, b)

print(c)
print("==================나눗셈==================")
def div1(a, b):
    return a/b

a = 1
b = 2
c = div1(a, b)

print(c)
print("==================뺄셈==================")
def sub1(a, b):
    return a-b

a = 1
b = 2
c = sub1(a, b)

print(c)

#매개변수가 없는 함수도 있다. (ex : a와 b , x혹은 y 등의 매개변수가 없다)
def sayYeh():
    return ' hi'
    
aaa= sayYeh()
print(aaa)


def sum1(a, b, c):
    return a+b+c

a = 1
b = 2
c = 34
d = sum1(a, b, c)
print(d)

