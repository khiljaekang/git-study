import tensorflow as tf
print(tf.__version__)

hello = tf.constant("hello world")

print(hello)              #그냥 print하면 hello의 자료형이 나온다

sess = tf.Session()
print(sess.run(hello))