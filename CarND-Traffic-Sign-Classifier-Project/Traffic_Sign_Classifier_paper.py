
import pickle
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# TODO: Fill this in based on where you saved the training and testing data

training_file = ".\\train.p"
validation_file= ".\\valid.p"
testing_file = ".\\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


import csv 
import pandas as pd
signnames=pd.read_csv('signnames.csv',header=[0])

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(signnames) 

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


import matplotlib.pyplot as plt

plt.imshow(X_train[1339])


def global_contrast_norm(image):
    # Reference : https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python
    import numpy as np
    import sys
    mu = np.mean(image)
    image = image - mu
    lda = 10
    s = 1
    contrast = np.sqrt(lda, np.mean(image**2))
    image = s*image / max(contrast,sys.float_info.epsilon)
    return image

def preprocess(imagelist):
    import numpy as np
    from skimage.color import rgb2yuv
    from sklearn import preprocessing

    yuv = rgb2yuv(imagelist)    
    global_norm = [global_contrast_norm(image[:,:,1]) for image in imagelist]
    norm = [np.atleast_3d(img) for img in norm]
    return norm


X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)

from sklearn.utils import shuffle

X_train,y_train = shuffle(X_train,y_train)

#plt.imshow(X_train[1339],cmap='gray')


import tensorflow as tf
from tensorflow.contrib.layers import flatten
EPOCHS = 50
BATCH_SIZE = 128

def LeNet(x):
    # Distribution parameters for random selection of weights and biases
    mu = 0
    sigma = .1

    # Layer 1 : Convolutional Layer Input = 32x32x1 Output = 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_W,strides=[1,1,1,1],padding='VALID') + conv1_b
    # Activation 
    conv1 = tf.nn.relu(conv1)
    # Pooling Input = 28x28x6 Output=14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # Layer 2 : Convolutional Layer Input = 14x14x6 Output= 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape = (5,5,6,16),mean = mu,stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1,conv2_W,strides = [1,1,1,1], padding = 'VALID') + conv2_b

    # Activation
    conv2 = tf.nn.relu(conv2)

    # Pooling Input = 10x10x16 Output = 5x5x16
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1], padding = 'VALID')

    # Flatten Input 5x5x16 Output = 400
    fc0 = flatten(conv2)

    # Layer 3 : Fully Connected Input = 400 Output = 120
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400,120),mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0,fc1_W) + fc1_b

    # ACtivation
    fc1 = tf.nn.relu(fc1)

    # Layer 4 : Fully Connected Input = 120 Output = 84
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120,84),mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_W) + fc2_b

    # ACtivation
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5 : Output : Fully Connected Input = 84 Output = 43
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84,43),mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2,fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32,(None,32,32,1))
y = tf.placeholder(tf.int32,(None))
one_hot_y = tf.one_hot(y,43)

rate = .001


logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y,logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver = tf.train.Saver()

def evaluate(X_data,y_data):
    num_samples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,num_samples,BATCH_SIZE):
        batch_x,batch_y =X_data[offset:offset+BATCH_SIZE],y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation,feed_dict={x:batch_x,y:batch_y})
        total_accuracy += (accuracy*len(batch_x))
    return total_accuracy/num_samples



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_samples = len(X_train)
    print('Training ...')
    print()
    for i in range(EPOCHS):
        X_train,y_train = shuffle(X_train,y_train)
        for offset in range(0,num_samples,BATCH_SIZE):
            end=offset+BATCH_SIZE
            batch_x,batch_y = X_train[offset:end],y_train[offset:end]
            sess.run(training_operation,feed_dict = {x:batch_x,y:batch_y})

        validation_accuracy = evaluate(X_valid,y_valid)
        print("EPOCH {} ....".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    saver.save(sess,'.traffic_lenet')
    print('Model saved')



# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


