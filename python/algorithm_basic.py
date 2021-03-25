#더하기
def sum(n,S):
    result = 0
    for i in range(1, n + 1):
        result = result + S[i]
    return result

S = [-1, 10, 7, 11, 5, 13, 8]
sum = sum(len(S) - 1, S)
print('sum =', sum) 

#정렬

#문제 : N개의 수로 구성된 리스트 S를 비내림차순으로 정렬하시오.
#해답 : S를 비내림차순으로 정렬한 리스트 
#파라미터: S , n 
#입력사례: S = [-1, 5, 7, 8, 10, 11, 13]
#교환정렬, 삽입정렬 선택정렬, 합병정렬, 퀵정렬 등 

#교환정렬(Exchange Sort)

def exchange(S):
    n = len(S)
    for i in range(n - 1):   # ( 0 , n-2)
        for j in range(i + 1, n):   #(i+1 , n-1)
            if (S[i] > S[j]):
                S[i], S[j] = S[j], S[i]    #swap   
 
