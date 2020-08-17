import os

from keras.layers.core import Dropout
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data import ImportImageData
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def addBasic(model):
    # Basic Network
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

def addLeNet(model):
    # LeNet
    model.add(Convolution2D(6,5,5, activation ='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation ='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.summary()
    return model

def addNVIDIACNN(model):
    model.add(Convolution2D(24,5,5, activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Convolution2D(36,5,5, activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Convolution2D(48,5,5, activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Convolution2D(64,3,3, activation ='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model

def addNVIDIACNN_WithoutPooling(model):
    model.add(Convolution2D(24,5,5, activation ='relu',subsample=(2,2)))
    model.add(Convolution2D(36,5,5, activation ='relu',subsample=(2,2)))
    model.add(Convolution2D(48,5,5, activation ='relu',subsample=(2,2)))
    model.add(Convolution2D(64,3,3, activation ='relu'))
    model.add(Convolution2D(64,3,3, activation ='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation ='relu'))
    model.add(Dropout(.2))
    model.add(Dense(50, activation ='relu'))
    model.add(Dense(10, activation ='relu'))
    model.add(Dense(1))
    model.summary()
    return model



# datafolders = ['..\\..\\Track1\\New\\Center\\driving_log.csv','..\\..\\Track1\\New\\Reverse\\driving_log.csv','..\\..\\Track1\\New\\Curves\\driving_log.csv']

datafolders = ['..\\..\\Track1\\New\\hugleft\\driving_log.csv','..\\..\\Track1\\New\\Reverse\\driving_log.csv','..\\..\\Track1\\New\\edges\\driving_log.csv','..\\..\\Track1\\New\\bad\\driving_log.csv']
# datafolders = ['..\\..\\Track1\\New\\hugleft\\driving_log.csv','..\\..\\Track1\\New\\bad\\driving_log.csv']
augmented_images,augmented_measurements = [], []
correction = 0.2

for folder in datafolders:
    c,l,r,st = ImportImageData(folder)
    for image,limage,rimage,sangle in zip(c,l,r,st):
        # Add middle
        augmented_images.append(image)
        augmented_measurements.append(sangle)
        # Add left
        augmented_images.append(limage)
        augmented_measurements.append(sangle+correction)
        # Add right
        augmented_images.append(rimage)
        augmented_measurements.append(sangle-correction)
        # Add flipped
        flipped_image = cv2.flip(image,1) # Flip Horizontal
        augmented_images.append(flipped_image)
        augmented_measurements.append(sangle*-1.0)


augmented_images = np.array(augmented_images)
augmented_measurements = np.array(augmented_measurements)

augmented_images,augmented_measurements = shuffle(augmented_images, augmented_measurements)
X_train, X_test, y_train, y_test = train_test_split(augmented_images, augmented_measurements, test_size=0.33)

def create_generator(data, batch_size):
    total = len(data)
    while 1:
        for idx in range(0,total,batch_size):
            subset = data[idx:idx+batch_size]
            


model = Sequential()
# Pre-Processing Layers
model.add(Lambda(lambda x : x-255.0 - 0.5, input_shape=(160,320,3))) # Normalization
model.add(Cropping2D(cropping=((70,25),(0,0)))) # Cropping
# Network
model = addNVIDIACNN_WithoutPooling(model)

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=.25, shuffle = True, nb_epoch = 5)
# Save out results
model.save('model_nvidia.h5')



