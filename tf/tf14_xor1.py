import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

# x, y, w, b hypothesis, cost, train
# sigmoid 사용
# predict, accuracy 준비해 놓을 것.

x = tf.placeholder(tf.float32, shape= [None, 2])
y = tf.placeholder(tf.float32, shape= [None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
                                # x의 열의 값과 동일 해야 함(행렬 계산) 

b = tf.Variable(tf.random_normal([1]), name = 'bias') 

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) #wx + b   matmul은 행렬곱


# cost =  tf.reduce_mean(tf.square( hypothesis - y))

cost = -tf.reduce_mean(y *tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis)) #시그모이드에 대한 코스트 정의


optimizer = tf.train.GradientDescentOptimizer(learning_rate= 4.9e-2) 
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# 프레딕트랑 에큐러시를 만들어도 새션안에서 런을 해줘야 실행을 되는거다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5000):
         cost_val, _= sess.run([cost, train],
                                feed_dict = {x: x_data, y: y_data})
    if step % 200 == 0 :
        print(step, cost_val)
    
    h, c, a = sess.run([hypothesis,predicted, accuracy],
                       feed_dict={x:x_data, y:y_data})

    print("\n Hypothesis : ", h, "\n Correct(y) : ", c,
          "\n Accuracy : ", a)
