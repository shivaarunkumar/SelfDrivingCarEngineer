# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.
# %% [markdown]
# ---
# ## Step 0: Load The Data

# %%
# Load pickled data
import pickle
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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

# %% [markdown]
# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 
# %% [markdown]
# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# %%
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

import csv 
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# %% [markdown]
# ### Include an exploratory visualization of the dataset
# %% [markdown]
# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# %%
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import numpy as np
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

# Histogram of the different classes present in the training data set
unique_classes,indices,counts = np.unique(y_train,return_index=True,return_counts=True)
plt.bar(unique_classes,counts)
plt.xlabel('Class ID')
plt.ylabel('Number of training sets')
avg_count = np.mean(counts)
print("Average Counts per class : " , avg_count)
plt.hlines(avg_count,0,n_classes,colors='r',linestyles='solid')
print(np.min(counts))
# Visualize classes of images

f,ax = plt.subplots(1,n_classes,figsize=(50,20))
i=0
for cid,cindex,count in zip(unique_classes,indices,counts):
    ax[i].set_title(str(cid)) #+ " " + signnames.values[:,1][cid]
    ax[i].imshow(X_train[cindex+np.random.randint(0,count)])
    i+=1

    

# %% [markdown]
# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# %% [markdown]
# ### Pre-process the Data Set (normalization, grayscale, etc.)
# %% [markdown]
# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# %%
import numpy as np
from skimage.color import rgb2gray
from sklearn import preprocessing
import cv2

def global_contrast_norm(image):
    # Reference : https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python
    import numpy as np
    import sys
    mu = np.mean(image)
    image = image - mu
    lda = 10
    s = 1
    contrast = np.sqrt(lda+ np.mean(image**2))
    image = s*image / max(contrast,sys.float_info.epsilon)
    return image
img = X_train[np.random.randint(0,len(X_train))]
ychan = cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2YUV))[0]
norm =global_contrast_norm(ychan) 

f,ax = plt.subplots(1,3,figsize=(10,2))
ax[0].imshow(img)
ax[0].set_title('original')
ax[1].imshow(ychan,cmap='gray')
ax[1].set_title('grayscale')
ax[2].imshow(norm,cmap='gray')
ax[2].set_title('normalized')


# %%
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
def preprocess(imagelist):
    import numpy as np
    from skimage.color import rgb2gray
    from sklearn import preprocessing

    ychan = [cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2YUV))[0] for img in imagelist]
    norm = [global_contrast_norm(image) for image in ychan]
    norm = [np.atleast_3d(img) for img in norm]
    return norm


X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)



# %% [markdown]
# ### Data Augmentation
# 
# As observed from the histogram of available samples, there is considerable imbalance in the available data that could lead to unforeseen bias when training. In order to balance the data set it is important to employ data augmentation for the training data set.
# 
# * [Reference 1](https://www.tensorflow.org/tutorials/images/data_augmentation)
# * [Reference 2](https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec)
# * [Reference 3](https://www.youtube.com/watch?v=UGiLdf3fzAI)
# 
# 

# %%
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import cv2

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def random_translate(image_array: ndarray):
    npixels = np.random.randint(1,5)
    tx = np.random.choice(range(-npixels, npixels))
    ty = np.random.choice(range(-npixels, npixels))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return np.atleast_3d(cv2.warpAffine(src=image_array, M=M, dsize=(32, 32)))


available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'translate' : random_translate    
}

target_samples = 3000

for cid,cindex,count in zip(unique_classes,indices,counts):
    required = target_samples - count
    augmented = []
    for i in range(required):
        random_id=cindex+np.random.randint(0,count)
        image_to_transform=X_train[random_id]
        transformations_to_apply = random.randint(1, len(available_transformations))
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        
        augmented.append(transformed_image)
    X_train =np.append(X_train,augmented,axis=0)
    y_train = np.append(y_train,[cid]*required)

# %% [markdown]
# ### Model Architecture

# %%
#%%
from sklearn.utils import shuffle

X_train,y_train = shuffle(X_train,y_train)


### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from tensorflow.contrib.layers import flatten
EPOCHS = 10
BATCH_SIZE = 128

def LeNet(x):
    # Distribution parameters for random selection of weights and biases
    mu = 0
    sigma = .1

    # Layer 1 : Convolutional Layer Input = 32x32x1 Output = 28*28*16
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,1,16),mean=mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1 = tf.nn.conv2d(x,conv1_W,strides=[1,1,1,1],padding='VALID') + conv1_b
    # Activation 
    conv1 = tf.nn.relu(conv1)
    # Pooling Input = 28x28x16 Output=14x14x16
    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # Layer 2.1 : Convolutional Layer Input = 14*14*16 Output= 10*10*24
    conv21_W = tf.Variable(tf.truncated_normal(shape = (5,5,16,24),mean = mu,stddev = sigma))
    conv21_b = tf.Variable(tf.zeros(24))
    conv21 = tf.nn.conv2d(pool1,conv21_W,strides = [1,1,1,1], padding = 'VALID') + conv21_b

    # Activation
    conv21 = tf.nn.relu(conv21)

    # Layer 2.2 : Convolutional Layer Input = 14*14*16 Output= 8*8*32
    conv22_W = tf.Variable(tf.truncated_normal(shape = (7,7,16,32),mean = mu,stddev = sigma))
    conv22_b = tf.Variable(tf.zeros(32))
    conv22 = tf.nn.conv2d(pool1,conv22_W,strides = [1,1,1,1], padding = 'VALID') + conv22_b

    # Activation
    conv22 = tf.nn.relu(conv22)

    # Pooling Input = 8*8*32 Output = 4*4*32
    pool22 = tf.nn.max_pool(conv22,ksize=[1,2,2,1],strides=[1,2,2,1], padding = 'VALID')

    # Flatten Input 10*10*24 + 4*4*32 Output = 2912
    fc01 = flatten(conv21)
    fc02 = flatten(pool22)
    fc0 = tf.concat([fc01, fc02], axis=1)
    # Layer 3 : Fully Connected Input = 2912 Output = 512
    fc1_W = tf.Variable(tf.truncated_normal(shape=(2912,512),mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1 = tf.matmul(fc0,fc1_W) + fc1_b

    # ACtivation
    fc1 = tf.nn.relu(fc1)

    # Layer 4 : Fully Connected Input = 512 Output = 256
    fc2_W = tf.Variable(tf.truncated_normal(shape=(512,256),mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(256))
    fc2 = tf.matmul(fc1,fc2_W) + fc2_b

    # ACtivation
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5 : Output : Fully Connected Input = 256 Output = 43
    fc3_W = tf.Variable(tf.truncated_normal(shape=(256,43),mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2,fc3_W) + fc3_b

    return logits, conv1,conv21,conv22


x = tf.placeholder(tf.float32,(None,32,32,1))
y = tf.placeholder(tf.int32,(None))
one_hot_y = tf.one_hot(y,43)

rate = .001
logits,conv1,conv21,conv22 = LeNet(x)
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
    saver.save(sess,'.traffic_lenet_enhanced')
    print('Model saved')


# %%
# Run against test dataset
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    
    print("Test Accuracy = {:.3f}".format(test_accuracy))

# %% [markdown]
# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.
# 
# * Load and Output the Images
# * Load the images and plot them here.
# * Feel free to use as many code cells as needed.
# * Predict the Sign Type for Each Image

# %%
import glob
import cv2
import matplotlib.image as mpimg
import os
flist = glob.glob('.\\internet_samples\\*.png')
fnames = [os.path.basename(f) for f in flist]
labels = [int(os.path.splitext(fname)[0]) for fname in fnames]
orig_images = [mpimg.imread(f) for f in flist]
orig_images = [img[:,:,:3] for img in orig_images]
print(orig_images[0].shape)
images = preprocess(np.array(orig_images))
plt.tight_layout()
f,ax = plt.subplots(1,len(images),figsize=(20,30))
for i in range(len(orig_images)):
    ax[i].imshow(orig_images[i])
    ax[i].text(-2, -2, str(labels[i])+":"+signnames.values[:,1][labels[i]], style='italic', bbox={'facecolor': 'deepskyblue', 'alpha': 1, 'pad': 2})


# %%
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
### Check Accuracy


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(images, labels)
    
    print("Test Accuracy = {:.3f}".format(test_accuracy))

# %% [markdown]
# ### Analyze Performance

# %%


softmax = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax,k=5)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./.traffic_lenet_enhanced.meta')
    saver.restore(sess, "./.traffic_lenet_enhanced")
    top_results = sess.run(top_k,feed_dict={x:images,y:labels})
    predictions = sess.run(softmax,feed_dict={x:images,y:labels})
    

f,ax = plt.subplots(len(images),2,figsize=(20,30))

for i in range(len(images)):
    ax[i][0].imshow(orig_images[i])
    ax[i][0].text(-2, -2, str(labels[i])+":"+signnames.values[:,1][labels[i]], style='italic', bbox={'facecolor': 'deepskyblue', 'alpha': 1, 'pad': 2})
    ax[i][1].bar(range(n_classes),predictions[i])
    ax[i][1].set_ylabel('Soft-max probability')
    ax[i][1].set_title("Predicted class :"+ str(np.argmax(predictions[i])))
plt.tight_layout()

# %% [markdown]
# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web
# %% [markdown]
# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# %%
print(top_results)

# %% [markdown]
# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 
# %% [markdown]
# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
# %% [markdown]
# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# %%
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

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


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./.traffic_lenet.meta')
    saver.restore(sess, "./.traffic_lenet")
    id = np.random.randint(0,len(orig_images))
    img = np.expand_dims(images[id], axis=0)
    
    plt.figure(figsize=(20,30))
    outputFeatureMap(img, conv1, plt_num=1)
    outputFeatureMap(img, conv2, plt_num=2)
    plt.tight_layout()
    plt.figure()
    plt.imshow(orig_images[id])
    plt.suptitle(signnames.values[:,1][labels[id]])


# %%



