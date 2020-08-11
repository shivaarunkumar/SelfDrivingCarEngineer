import pickle
import numpy as np
from skimage.color import rgb2gray
from sklearn import preprocessing
import cv2
import numpy as np
import sys
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

training_file = ".\\train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']



def global_contrast_norm(image):
    # Reference : https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python
    
    mu = np.mean(image)
    image = image - mu
    lda = 10
    s = 1
    contrast = np.sqrt(lda+ np.mean(image**2))
    image = s*image / max(contrast,sys.float_info.epsilon)
    return image

def preprocess(imagelist):
    ychan = [cv2.split(cv2.cvtColor(img,cv2.COLOR_RGB2YUV))[0] for img in imagelist]
    norm = [global_contrast_norm(image) for image in ychan]
    norm = [np.atleast_3d(img) for img in norm]
    return norm


X_train = preprocess(X_train)
unique_classes,indices,counts = np.unique(y_train,return_index=True,return_counts=True)


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

print(len(X_train))
print(len(y_train))