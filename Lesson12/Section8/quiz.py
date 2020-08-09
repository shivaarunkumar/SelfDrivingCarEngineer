# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Solution is available in the other "solution.py" tab
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]


# %%
# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]


# %%
# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])


# %%
# TODO: Create Model
hidden_logits = tf.add(tf.matmul(features,weights[0]),biases[0])
hidden_output = tf.nn.relu(hidden_logits)

logits = tf.add(tf.matmul(hidden_output,weights[1]),biases[1])


# %%
# TODO: save and print session results on a variable named "output"
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    output = session.run(logits)
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
