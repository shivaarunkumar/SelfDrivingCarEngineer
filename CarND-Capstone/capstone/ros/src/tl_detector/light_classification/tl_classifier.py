from styx_msgs.msg import TrafficLight
import os
import logging
from keras.models import Model, model_from_yaml
import tensorflow as tf
import numpy as np
import cv2
import rospy

class TLClassifier(object):
    def __init__(self):
        # load classifier
        rospy.loginfo("Loading Classifier")
        with open('light_classification/resnet50_model_struc.yaml') as yaml_file:
            classifier_struct = yaml_file.read()

        self.classifier_iv3 = model_from_yaml(classifier_struct)
        self.classifier_iv3.load_weights('light_classification/resnet50_model.h5')
        self.tf_graph = tf.get_default_graph()
        self.label_map = {0: TrafficLight.RED, 1: TrafficLight.YELLOW, 2: TrafficLight.GREEN, 3: TrafficLight.UNKNOWN}
        # self.classifier_iv3.summary()
        rospy.loginfo("Classifier loaded, start simulator")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        prediction = 3
        # implement light color prediction
        with self.tf_graph.as_default():
            prediction = np.argmax(self.classifier_iv3.predict(cv2.resize(image, (224, 224)).reshape(1, 224, 224, 3))[0])
        #rospy.loginfo(prediction)
        return self.label_map[int(prediction)]
