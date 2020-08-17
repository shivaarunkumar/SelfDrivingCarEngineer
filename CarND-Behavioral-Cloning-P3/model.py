import os
import cv2
import csv
import matplotlib.image as mpimg
import numpy as np
from math import ceil

from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from tqdm import tqdm


#vscode requirements related to root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# List of folders to parse for images
imgdatafiles = ['..\\..\\data\\driving_log.csv',
                #'..\\..\\Track1\\New\\hugleft\\driving_log.csv',
                #'..\\..\\Track1\\New\\Reverse\\driving_log.csv',
                '..\\..\\Track1\\New\\Reverse2\\driving_log.csv',
                #'..\\..\\Track1\\New\\TheCurve\\driving_log.csv',
                #'..\\..\\Track1\\New\\edges\\driving_log.csv',
                #'..\\..\\Track1\\New\\bad2\\driving_log.csv',
                #'..\\..\\Track1\\New\\bad\\driving_log.csv'
                ]

# Dataset Initializers
images = []
stangles = []
correction = 0.2

# Parse
for imgdatafile in imgdatafiles:
    print('Processing : '+imgdatafile)
    dirname = os.path.dirname(imgdatafile)
    imgfolder = 'IMG'
    lines = []
    with open(imgdatafile,'r') as datafile:
        reader = csv.reader(datafile)
        for line in reader:
            lines.append(line)
            
    with tqdm(total=len(lines)) as pbar:
        for line in lines:
            pbar.update(1)
            (center,left,right,sangle,throttle,breakst,speed)=line
            sangle = float(sangle)
            center = os.path.join(dirname,imgfolder,os.path.basename(center))
            images.append(center)
            stangles.append(sangle)            
            left = os.path.join(dirname,imgfolder,os.path.basename(center))
            images.append(left)
            stangles.append(sangle+correction)
            right = os.path.join(dirname,imgfolder,os.path.basename(center))
            images.append(left)
            stangles.append(sangle-correction)

images,stangles = shuffle(images, stangles)
X_train, X_val, y_train, y_val = train_test_split(images, stangles, test_size=0.2)

# Batch and Generate
def create_generator(x,y, batch_size = 32):
    total = len(x)
    while 1:
        for idx in range(0,total,batch_size):
            x_subset = x[idx:idx+batch_size]
            y_subset = y[idx:idx+batch_size]
            images = []
            angles = []
            for i in range(len(y_subset)):
                images.append(mpimg.imread(x_subset[i]))
                angles.append(y_subset[i])
            images=np.array(images)
            angles=np.array(angles)

            yield shuffle(images,angles)

batch_size = 32
training_generator = create_generator(X_train, y_train,batch_size)
validation_generator = create_generator(X_val, y_val,batch_size)


# NVIDIA Model Definition
def addNVIDIACNN(model):
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

model = Sequential()
# Pre-Processing Layers
model.add(Lambda(lambda x : x/127.5 - 1, input_shape=(160,320,3))) # Normalization x-255.0 - 0.5
model.add(Cropping2D(cropping=((70,25),(0,0)))) # Cropping
# Network
model = addNVIDIACNN(model)

model.compile(loss='mse', optimizer='adam')

model.fit_generator(training_generator, 
            steps_per_epoch=ceil(len(X_train)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(X_val)/batch_size), 
            epochs=3)
# Save out results
model.save('model_nvidia.h5')