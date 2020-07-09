from sklearn.datasets import load_breast_cancer
import tensorflow as tf

cancer = load_breast_cancer()

x_data = cancer.data
y_data = cancer.target


x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.zeros([30, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name= 'bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) #wx + b   matmul은 행렬곱


# cost =  tf.reduce_mean(tf.square( hypothesis - y))

cost = -tf.reduce_mean(y *tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis)) #시그모이드에 대한 코스트 정의


optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-10) 
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# 프레딕트랑 에큐러시를 만들어도 새션안에서 런을 해줘야 실행을 되는거다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20000):
         cost_val, _= sess.run([cost, train],
                                feed_dict = {x: x_data, y: y_data})
    if step % 200 == 0 :
        print(step, cost_val)
    
    h, c, a = sess.run([hypothesis,predicted, accuracy],
                       feed_dict={x:x_data, y:y_data})

    print( "\n Hypothesis : ", h,"\n Correct(y) : ", c,
           "\n Accuracy : ", a,)