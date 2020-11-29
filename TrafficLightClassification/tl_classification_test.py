import os
import logging
from keras.models import Model, model_from_yaml
import tensorflow as tf
import numpy as np
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open('resnet50_model_struc.yaml') as yaml_file:
    classifier_struct = yaml_file.read()
classifier = model_from_yaml(classifier_struct)
classifier.load_weights('resnet50_model.h5')
classifier.summary()

graph = tf.get_default_graph()
graph.as_default()
image1 = cv2.resize(cv2.imread('test_images\\green.jpg'), (224, 224)).reshape(1, 224, 224, 3)
image2 = cv2.resize(cv2.imread('test_images\\red.jpg'), (224, 224)).reshape(1, 224, 224, 3)
image3 = cv2.resize(cv2.imread('test_images\\daySequence1--00081.jpg'), (224, 224)).reshape(1, 224, 224, 3)
image4 = cv2.resize(cv2.imread('test_images\\daySequence1--00316.jpg'), (224, 224)).reshape(1, 224, 224, 3)
image5 = cv2.resize(cv2.imread('test_images\\test\\frame0066.jpg'), (224, 224)).reshape(1, 224, 224, 3)
image6 = cv2.resize(cv2.imread('test_images\\test\\frame0010.jpg'), (224, 224)).reshape(1, 224, 224, 3)

print(classifier.predict(image1))
print(classifier.predict(image2))
print(classifier.predict(image3))
print(classifier.predict(image4))
print(classifier.predict(image5))
print(classifier.predict(image6))

with open('iv3_withouttop_struc.yaml') as yaml_file:
    classifier_struct = yaml_file.read()
classifier_iv3 = model_from_yaml(classifier_struct)
classifier_iv3.load_weights('iv3_model_withouttop.h5')
classifier_iv3.summary()

print(classifier_iv3.predict(image1))
print(classifier_iv3.predict(image2))
print(classifier_iv3.predict(image3))
print(classifier_iv3.predict(image4))
print(classifier_iv3.predict(image5))
print(classifier_iv3.predict(image6))

print(int(np.argmax(classifier_iv3.predict(image4))))