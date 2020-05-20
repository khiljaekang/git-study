#3. 딕셔너리 #중복 X
# {키 : 벨류}
# {key :value}

a = {1: 'hi', 2:'hello'}
print(a)
print(a[1])                                  #불러올때 [ ] 해줘야함
print(a[2])

b = {'hi': 1, 'hello': 2}                   
print(b['hello'])

# 딕셔너리 요소 삭제
del a[1]
print (a)
del a[2]
print (a)                                    #모두 지워버리면 안에 내용없이 딕셔너리 형이라는 것만 출력

a = {1:'a', 1:'b', 1:'c'}
print(a)                                     # key는 중복이 안되므로 덥혀씌워진 마지막  'c'만 남는다.

b = {1:'a', 2:'a', 3: 'a'}
print(b)                                     #value는 중복가능 ex)셋 다 100점 맞음

a = {'name' : 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys())
print(a.values())
print(type(a))
print(a.get('name'))                         # 출력값이 같다
print(a['name'])                             # 출력값이 같다
print(a.get('phone'))
print(a['phone'])



