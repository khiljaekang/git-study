'''
test_list = ['one', 'two', 'three',]
for i in test_list:
    print(i)

# a = [(1,2),(3,4),(5,6)]
# for (first, last) in a:
#     print(first+last)

'''

# hi = "안녕하세요"
# for s in hi:
#     print(s)

'''
score_list = [90, 45, 70, 60, 55]
number = 1 
for score in score_list:
    if score >= 60:
        print("{}번 학생은 합격 입니다".format(number))
    else:
        print("{}번 학생은 불합격 입니다".format(number))   
    number +=1 
'''
    

score_list = [90, 45, 70, 60, 55]
number = 1 
for score in score_list:
    if score >= 60:
        result = "합격"
    else:
        result = "불합격" 
    print("{}번 학생은 {}입니다".format(number, result))   
    number +=1 
    