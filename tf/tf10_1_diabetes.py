from sklearn.datasets import load_diabetes
import tensorflow as tf

diabetes = load_diabetes()

x_data = diabetes.data
y_data = diabetes.target

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([10, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name= 'bias')

hypothesis = tf.matmul(x, w) + b #wx + b   matmul은 행렬곱


cost =  tf.reduce_mean(tf.square( hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 4.9e-1) #0.00001

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    cost_val, hy_val, _= sess.run([cost, hypothesis, train],
                                feed_dict = {x: x_data, y: y_data})
    if step % 10 == 0 :
        print(step,"\n 예측값: ", hy_val, 'cost :',cost_val, )
