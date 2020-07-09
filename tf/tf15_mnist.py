import tensorflow as tf
from keras.datasets import mnist 
from sklearn.metrics import accuracy_score

mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') /255  
x_test = x_test.reshape(10000, 28*28).astype('float32') /255. 

x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.zeros([28*28, 500]), name = 'weight1')                            
b1 = tf.Variable(tf.zeros([500]), name = 'bias1') 
layer1 = tf.matmul(x, w1) + b1 
#model.add(Dense(100, input_dim=2))
w2 = tf.Variable(tf.zeros([500, 400]), name = 'weight2')    
b2 = tf.Variable(tf.zeros([400]), name = 'bias2') 
layer2 = tf.matmul(layer1, w2) + b2                      
#model.add(Dense(50))
w3 = tf.Variable(tf.zeros([400, 300]), name = 'weight3')    
b3 = tf.Variable(tf.zeros([300]), name = 'bias3') 
layer3 = tf.matmul(layer2, w3) + b3  

w4 = tf.Variable(tf.zeros([300, 250]), name = 'weight3')    
b4 = tf.Variable(tf.zeros([250]), name = 'bias3') 
layer4 = tf.matmul(layer3, w4) + b4    

w5 = tf.Variable(tf.zeros([250, 200]), name = 'weight3')    
b5 = tf.Variable(tf.zeros([200]), name = 'bias3') 
layer5 = tf.matmul(layer4, w5) + b5       

w6 = tf.Variable(tf.zeros([200, 150]), name = 'weight3')    
b6 = tf.Variable(tf.zeros([150]), name = 'bias3') 
layer6 = tf.matmul(layer5, w6) + b6      

w7 = tf.Variable(tf.zeros([150, 100]), name = 'weight3')    
b7 = tf.Variable(tf.zeros([100]), name = 'bias3') 
layer7 = tf.matmul(layer6, w7) + b7       

w8 = tf.Variable(tf.zeros([100, 50]), name = 'weight3')    
b8 = tf.Variable(tf.zeros([50]), name = 'bias3') 
layer8 = tf.matmul(layer7, w8) + b8     

w9 = tf.Variable(tf.zeros([50, 30]), name = 'weight3')    
b9 = tf.Variable(tf.zeros([30]), name = 'bias3') 
layer9 = tf.matmul(layer8, w9) + b9       
#model.add(Dense(1))              

w10 = tf.Variable(tf.zeros([30, 10]), name = 'weight3')    
b10 = tf.Variable(tf.zeros([10]), name = 'bias3') 
hypothesis = tf.nn.softmax(tf.matmul(layer9, w10) + b10)      
       

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0000.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train, depth=10 ))  
    y_test = sess.run(tf.one_hot(y_test, depth=10 )) 
   

    for step in range(100):
        _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_train, y:y_train})

        if step % 10 ==0:
            print(step, cost_val)

    # 최적의 W와 b가 구해져 있다
    a = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = sess.run(tf.argmax(a, 1))
    print(a, y_pred )

    #1. Accuracy - sklearn
    y_pred = sess.run(tf.one_hot(y_pred, 10))
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy :', acc)

    #2. Accuracy - tensorflow
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))


