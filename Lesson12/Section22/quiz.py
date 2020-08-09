# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Solution is available in the other "solution.py"
import tensorflow as tf

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

tf.set_random_seed(123456)


hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], 
[11.0, 12.0, 13.0, 14.0]])


keep_prob = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features,weights[0]),biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer,keep_prob)

logits = tf.add(tf.matmul(hidden_layer,weights[1]),biases[1])


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(logits,feed_dict={keep_prob:.5})
    print(output)
# %% [markdown]
# ### Running the Grader
# 
# To run the grader below, you'll want to run the above training from scratch (if you have otherwise already ran it multiple times). You can reset your kernel and then run all cells for the grader code to appropriately check that you weights and biases achieved the desired end result.

# %%
### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(output)
except Exception as err:
    print(str(err))




# %%
