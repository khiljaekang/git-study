import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

dataset = np.loadtxt('./data/csv/data-01-test-score.csv', 
                       delimiter=',', dtype=np.float32)
x_data = dataset[:, 0:-1]
y_data = dataset[:, [-1]]






x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name= 'bias')

hypothesis = tf.matmul(x, w) + b #wx + b   matmul은 행렬곱


cost =  tf.reduce_mean(tf.square( hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 4.9e-5) 

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    cost_val, hy_val, _= sess.run([cost, hypothesis, train],
                                feed_dict = {x: x_data, y: y_data})
    if step % 10 == 0 :
        print(step, 'cost :',cost_val, "\n 예측값: ", hy_val)

# sess.close() # 큰 데이터를 사용할 시에는 사용 후 닫아주는 것이 좋다