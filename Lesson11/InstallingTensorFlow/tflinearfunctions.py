import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.Variable(5, name ="a")
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
bias = tf.Variable(tf.zeros(n_labels))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    b= a.assign(a+1)
    print(a.eval())
    print(b.eval())
    print(weights.get_shape())
    print(bias.eval())