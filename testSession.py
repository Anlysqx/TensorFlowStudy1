import tensorflow as tf

hello = tf.constant('hello tensorflow session')
sess = tf.Session()
print(sess.run(hello))
sess.close()
