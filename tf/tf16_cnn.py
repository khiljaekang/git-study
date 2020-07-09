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
# layer2 = tf.sigmoid(tf.matmul(x, w) + b)
# layer2 = tf.matmul(layer1, w2) + b2   
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)        
# layer2 = tf.nn.selu(tf.matmul(layer1, w2) + b2)                   
# layer2 = tf.nn.elu(tf.matmul(layer1, w2) + b2)                   

#model.add(Dense(50))
w3 = tf.Variable(tf.zeros([400, 300]), name = 'weight3')    
b3 = tf.Variable(tf.zeros([300]), name = 'bias3') 
layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3)  

w4 = tf.Variable(tf.zeros([300, 250]), name = 'weight4')    
b4 = tf.Variable(tf.zeros([250]), name = 'bias4') 
layer4 = tf.nn.elu(tf.matmul(layer3, w4) + b4)    

# w5 = tf.Variable(tf.zeros([250, 150]), name = 'weight4')    
# b5 = tf.Variable(tf.zeros([150]), name = 'bias4') 
# layer5 = tf.nn.dropout(layer4, keep_prob=0.1)  

w5 = tf.Variable(tf.zeros([250, 10]), name = 'weight5')    
b5 = tf.Variable(tf.zeros([10]), name = 'bias5') 
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5) 


  
       

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=00.1).minimize(cost)

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


