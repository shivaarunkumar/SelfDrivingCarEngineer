import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.constant([[1.,1.],[2.,2.]])

with tf.Session() as sess:
    print(tf.reduce_mean(x).eval())
    print(tf.reduce_mean(x,0).eval())
    print(tf.reduce_mean(x,1).eval())

