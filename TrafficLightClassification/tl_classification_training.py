# Generic
import os

# File I/O
from glob import glob
import yaml
from math import ceil
# Image Processing and Visualization
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
# Deep Learning
import tensorflow as tf
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.layers.core import Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Set current directory to notebook directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(globals()['_dh'][0])

# https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view
# Curated by ex udacity student Vatsal Srivastava :
# https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62
# Useful References: http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data

simulator_image_path = os.path.join('data', 'sim_training_data')
parkinglot_image_path = os.path.join('data', 'real_training_data')

# Alex's Dataset - Ex Udacity Student
# https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0
simulator_image_path2 = os.path.join('data', 'simulator_dataset_rgb')

# https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset
bosch_image_path = os.path.join('data', 'dataset_train_rgb')
bosch_image_path2 = os.path.join('data', 'dataset_test_rgb')

# https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset
lisa_daysequence1 = os.path.join('data', 'daySequence1')
lisa_daysequence2 = os.path.join('data', 'daySequence2')
lisa_daytrain = os.path.join('data', 'dayTrain')

# Import images and Labels
images = []
labels = []  # One hot encoded [red yellow green none]
bboxes = []
text = []
label_set = ['Red', 'Yellow', 'Green', 'None']
color_map = {'Green': (0, 255, 0), 'Red': (0, 0, 255), 'Yellow': (0, 255, 255)}
color_count = {'Green': 0, 'Red': 0, 'Yellow': 0, 'None': 0}
label_map = {'Green': [0, 0, 1, 0], 'Red': [1, 0, 0, 0], 'Yellow': [0, 1, 0, 0], 'None': [0, 0, 0, 1]}
n_classes = len(label_set)


def extract_image_labels(base_dir, yaml_file, path_tag='filename', annotation_tag='annotations',
                         label_tag='class', box_tags=None, points=False, rep_path=''):
    if box_tags is None:
        box_tags = ['xmin', 'ymin', 'x_width', 'y_height']
    for file in yaml_file:
        with open(file, 'r') as f:
            detection_data = yaml.load(f.read())
            f.close()
    with tqdm(total=len(detection_data)) as pbar:
        for data in detection_data:
            pbar.update(1)
            if rep_path == '':
                image_path = os.path.join(base_dir, data[path_tag])
            else:
                image_path = os.path.join(base_dir, rep_path, data[path_tag].split("/")[-1])
            label = [1, 1, 1, 1]
            for annotation in data[annotation_tag]:
                if annotation[label_tag] != 'off':
                    if annotation[label_tag].startswith('Green'):
                        annotation[label_tag] = 'Green'
                    elif annotation[label_tag].startswith('Red'):
                        annotation[label_tag] = 'Red'
                    else:
                        annotation[label_tag] = 'Yellow'
                    label = np.multiply(label, label_map[annotation[label_tag]])
                    label_text = annotation[label_tag]
                    if not points:
                        vertices = [tuple(map(int, (annotation[box_tags[0]], annotation[box_tags[1]]))),
                                    tuple(map(int, (
                                        annotation[box_tags[0]] + annotation[box_tags[2]],
                                        annotation[box_tags[1]] + annotation[box_tags[3]])))]
                    else:
                        vertices = [tuple(map(int, (annotation[box_tags[0]], annotation[box_tags[1]]))),
                                    tuple(map(int, (annotation[box_tags[2]], annotation[box_tags[3]])))]
                    bboxes.append(vertices)
            if not data[annotation_tag]:
                label = np.multiply(label, label_map['None'])
                label_text = 'None'
                vertices = []

            num_labels = len(np.where(label == 1)[0])
            if num_labels == 1:
                labels.append(label)
                images.append(image_path)
                bboxes.append(vertices)
                color_count[label_text] += 1


def plot_ground_truth(idx):
    image_path = images[idx]
    label = np.array(labels[idx])
    # bbox = bboxes[idx]
    image = cv2.imread(image_path)
    idx = np.where(label == 1)[0][0]
    label_text = label_set[idx]
    # if bbox:
    #     cv2.rectangle(image, bbox[0], bbox[1], color_map[label_text], 2)
    #     cv2.putText(image, label_text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, color_map[label_text], 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(label_text)
    plt.show()


# Simulator Images
extract_image_labels(simulator_image_path, glob(os.path.join(simulator_image_path, '*.yaml')))

# Extract Alex's Data
subfolders = [f.name for f in os.scandir(simulator_image_path2) if f.is_dir()]
for folder in subfolders:
    image_paths = glob(os.path.join(simulator_image_path2, folder, '*.png'))
    images.extend(image_paths)
    for i in range(len(image_paths)):
        labels.append(label_map[folder])
        bboxes.append([])
        color_count[folder] += 1

# Real Images
extract_image_labels(parkinglot_image_path, glob(os.path.join(parkinglot_image_path, '*.yaml')))

# Bosch Image Data Set
# extract_image_labels(bosch_image_path, glob(os.path.join(bosch_image_path, '*.yaml')),
#                      label_tag='label', path_tag='path', annotation_tag='boxes',
#                      box_tags=['x_min', 'y_min', 'x_max', 'y_max'], points=True)

extract_image_labels(bosch_image_path2, glob(os.path.join(bosch_image_path2, '*.yaml')),
                     label_tag='label', path_tag='path', annotation_tag='boxes',
                     box_tags=['x_min', 'y_min', 'x_max', 'y_max'], points=True, rep_path=os.path.join('rgb', 'test'))

print('Images : Total {} : Green : {} Red : {}  Yellow : {}  None : {}'.format(len(images), color_count['Green'],
                                                                               color_count['Red'],
                                                                               color_count['Yellow'],
                                                                               color_count['None']))
# plot_ground_truth(200)

images, labels = shuffle(images, labels)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)


# Batch and Generate
def create_generator(x, y, batch_size=32):
    total = len(x)
    while 1:
        for idx in range(0, total, batch_size):
            x_subset = x[idx:idx + batch_size]
            y_subset = y[idx:idx + batch_size]
            images_sub = []
            labels_sub = []
            for i in range(len(y_subset)):
                # print(x_subset[i])
                images_sub.append(cv2.resize(cv2.imread(x_subset[i]), (224, 224)))
                labels_sub.append(y_subset[i])
            images_sub = np.array(images_sub)
            labels_sub = np.array(labels_sub)

            yield shuffle(images_sub, labels_sub)


batch_size = 5


# training_generator = create_generator(X_train, y_train, batch_size)
# validation_generator = create_generator(X_val, y_val, batch_size)


# References
# https://keras.io/api/applications/
# https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
# https://keras.io/api/metrics/accuracy_metrics/#accuracy-class
# https://blog.paperspace.com/popular-deep-learning-architectures-resnet-inceptionv3-squeezenet/
def generate_model():
    inputs = Input(shape=(224, 224, 3))
    resnet_50 = ResNet50(weights='imagenet', input_tensor=inputs)
    output = resnet_50.output
    output = Dropout(.5, name="start_customization")(output)
    output = Dense(1024, activation='relu')(output)
    output = Dropout(.5)(output)
    predictions = Dense(n_classes, activation='softmax')(output)
    model = Model(inputs=inputs, outputs=predictions)
    return model


final_model = generate_model()
final_model.summary()
final_model.compile(loss="binary_crossentropy", optimizer='sgd',
                    metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
for i in range(5):
    final_model.fit_generator(create_generator(X_train, y_train, batch_size),
                              steps_per_epoch=ceil(len(X_train) / batch_size),
                              validation_data=create_generator(X_val, y_val, batch_size),
                              validation_steps=ceil(len(X_val) / batch_size),
                              epochs=10,
                              verbose=2)

# Save out results
final_model.save('resnet50_model.h5')
RESNET50_MODEL_WEIGHTS = os.path.join(os.getcwd(), 'resnet50_model_weights.h5')
RESNET50_MODEL_STRUCTURE = os.path.join(os.getcwd(), 'resnet50_model_struc.yaml')
with open(RESNET50_MODEL_STRUCTURE, "w") as struct_file:
    struct_file.write(final_model.to_yaml())


def generate_model_inceptionv3():
    inputs = Input(shape=(224, 224, 3))
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=True, input_tensor=inputs)

    # add a global spatial average pooling layer
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    x = Dropout(.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.5)(x)
    # and a logistic layer
    predictions = Dense(n_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inputs, outputs=predictions)
    # # first: train only the top layers (which were randomly initialized)
    # # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    return model


def generate_model_inceptionv3_withouttop():
    inputs = Input(shape=(224, 224, 3))
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=inputs)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(.5)(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(.5)(x)
    # and a logistic layer
    predictions = Dense(n_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inputs, outputs=predictions)
    # # first: train only the top layers (which were randomly initialized)
    # # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False
    return model


def generate_model_inceptionv3_withouttop_finetune():
    inputs = Input(shape=(224, 224, 3))
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=inputs)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(.5)(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(.5)(x)
    # and a logistic layer
    predictions = Dense(n_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inputs, outputs=predictions)
    # # first: train only the top layers (which were randomly initialized)
    # # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    return model


iv3_model = generate_model_inceptionv3()
iv3_model.summary()
iv3_model.compile(loss="binary_crossentropy", optimizer='rmsprop',
                  metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
for i in range(5):
    iv3_model.fit_generator(create_generator(X_train, y_train, batch_size),
                            steps_per_epoch=ceil(len(X_train) / batch_size),
                            validation_data=create_generator(X_val, y_val, batch_size),
                            validation_steps=ceil(len(X_val) / batch_size),
                            epochs=10,
                            verbose=2)

# Save out results
iv3_model.save('iv3_model.h5')
IV3_MODEL_WEIGHTS = os.path.join(os.getcwd(), 'iv3_model_weights.h5')
IV3_MODEL_STRUCTURE = os.path.join(os.getcwd(), 'iv3_struc.yaml')
with open(IV3_MODEL_STRUCTURE, "w") as struct_file:
    struct_file.write(iv3_model.to_yaml())

iv3_model = generate_model_inceptionv3_withouttop()
iv3_model.summary()
iv3_model.compile(loss="binary_crossentropy", optimizer='rmsprop',
                  metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
for i in range(5):
    iv3_model.fit_generator(create_generator(X_train, y_train, batch_size),
                            steps_per_epoch=ceil(len(X_train) / batch_size),
                            validation_data=create_generator(X_val, y_val, batch_size),
                            validation_steps=ceil(len(X_val) / batch_size),
                            epochs=10,
                            verbose=2)

# Save out results
iv3_model.save('iv3_model_withouttop.h5')
IV3_MODEL_WEIGHTS = os.path.join(os.getcwd(), 'iv3_model_withouttop_weights.h5')
IV3_MODEL_STRUCTURE = os.path.join(os.getcwd(), 'iv3_withouttop_struc.yaml')
with open(IV3_MODEL_STRUCTURE, "w") as struct_file:
    struct_file.write(iv3_model.to_yaml())

iv3_model = generate_model_inceptionv3_withouttop_finetune()
iv3_model.summary()
iv3_model.compile(loss="binary_crossentropy", optimizer='rmsprop',
                  metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
for i in range(5):
    iv3_model.fit_generator(create_generator(X_train, y_train, batch_size),
                            steps_per_epoch=ceil(len(X_train) / batch_size),
                            validation_data=create_generator(X_val, y_val, batch_size),
                            validation_steps=ceil(len(X_val) / batch_size),
                            epochs=10,
                            verbose=2)

# Save out results
iv3_model.save('iv3_model_finetune.h5')
IV3_MODEL_WEIGHTS = os.path.join(os.getcwd(), 'iv3_model_finetune_weights.h5')
IV3_MODEL_STRUCTURE = os.path.join(os.getcwd(), 'iv3_finetune_struc.yaml')
with open(IV3_MODEL_STRUCTURE, "w") as struct_file:
    struct_file.write(iv3_model.to_yaml())
