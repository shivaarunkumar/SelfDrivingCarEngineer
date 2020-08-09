import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.int32)

with tf.Session() as sess:
    output = sess.run(tf.subtract(tf.cast(tf.divide(x,y),tf.int32),tf.constant(1)),feed_dict = {x:10,y:2})
    