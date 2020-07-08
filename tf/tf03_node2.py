# 3 + 4 + 5 
# 4 - 2
# 3 * 4
# 4 / 2 

import tensorflow as tf
node1 = tf.constant(1.0, tf.float32)
node2 = tf.constant(2.0, tf.float32)
node3 = tf.constant(3.0, tf.float32)
node4 = tf.constant(4.0, tf.float32)
node5 = tf.constant(5.0, tf.float32)


# node10 = tf.add_n([node3,node4,node5])
node10 = tf.add(tf.add(node3,node4),node5)
node11 = tf.subtract(node4,node2)
node12 = tf.multiply(node3,node4)
node13 = tf.divide(node4,node2)

sess = tf.Session()

print("sess.run(node10) : ", sess.run([node10]))
print("sess.run(node11) : ", sess.run([node11]))
print("sess.run(node12) : ", sess.run([node12]))
print("sess.run(node13) : ", sess.run([node13]))


'''
tf.add (덧셈) 

tf.sub (뺄셈) 

tf.mul (곱셈) 

tf.div (나눗셈의 몫) 

tf.mod (나눗셈의 나머지) 

tf.abs (절댓값을 리턴) 

tf.neg (음수를 리턴)

tf.sign (부호를 리턴) (음수는 -1, 양수는 1, 0은 0)

tf.inv (역수를 리턴) (예: 3의 역수는 1/3)

tf.square (제곱을 계산) 

tf.round (반올림 값을 리턴) 

tf.sqrt (제곱근을 계산) 

tf.pow (거듭제곱 값을 계산) 

tf.exp (지수 값을 계산) 

tf.log (로그 값을 계산) 

tf.maximum (최댓값을 리턴) 

tf.minimum (최솟값을 리턴) 

tf.cos (코사인 함수 값을 계산) 

tf.sin (사인 함수 값을 계산) 

tf.diag (대각행렬을 리턴)

tf.transpose (전치행렬을 리턴)

tf.matmul (두 텐서를 행렬곱한 결과 텐서를 리턴)

tf.matrix_determinant (정방행렬의 행렬식 값을 리턴)

tf.matrix_inverse (정방행렬의 역행렬을 리턴)

'''