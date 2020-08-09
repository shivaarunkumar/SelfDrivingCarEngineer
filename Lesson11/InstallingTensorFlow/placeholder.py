import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict = {x : 'Hello World!'})
    y = tf.add(5,7)
    print(output) 
    print(y.eval())