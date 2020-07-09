import tensorflow as tf
from keras.datasets import mnist 
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical

mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') /255  
x_test = x_test.reshape(10000, 28*28).astype('float32') /255. 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)       #(60000,784)  (10000,784)
print(y_train.shape, y_test.shape)       #(60000,10)  (10000,10) 

learing_rate = 0.01
traing_epoch = 15
batch_size = 100
total_batch = int(len(x_train)/ batch_size)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)        #dropout

# w1 =tf.Variable(tf.random_normal([784, 512]), name="weight")
w1 = tf.get_variable("w1", shape=[784, 512],
                      initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

w2 = tf.get_variable("w2", shape=[512, 512],
                      initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

w3 = tf.get_variable("w3", shape=[512, 512],
                      initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)

w4 = tf.get_variable("w4", shape=[512, 256],
                      initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

w5 = tf.get_variable("w5", shape=[256, 10],
                      initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
L5 = tf.nn.selu(tf.matmul(L4, w5) + b5)
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)      
       

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(traing_epoch):      # 15
    ########################이부분 구현할 것 #####################
    ave_cost = 0

    for i in range(total_batch):       # 600
          batch_xs, batch_ys = x_train([batch_size])
          feed_dict = {x: batch_xs, y:batch_ys, keep_prob:0.7}
          c, _ = sess.run([cost,optimizer], feed_dict=feed_dict)
          avg_cost += c/ total_batch

    print('Epoch: ', '%04d' %(epoch + 1),
          'cost = ', '{:.9f'}.format(avg_cost))
print('훈련 끝!!') 

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc: ')  ### acc 출력할 것 