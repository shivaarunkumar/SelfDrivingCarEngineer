import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
from keras.models import load_model
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Visualize
image = mpimg.imread('..\\..\\data\\IMG\\center_2016_12_01_13_45_34_674.jpg')
model = load_model('model_nvidia.h5')
model.summary()
plt.figure()
plt.imshow(image)
plt.suptitle('Original Image')

normalize_layer = model.get_layer(index=0)
crop_layer = model.get_layer(index=1)

normalized_output = K.function([normalize_layer.input],[normalize_layer.output])
normalized = normalized_output([image[None,...]])[0]
plt.figure()
plt.imshow(normalized[0])
#plt.imshow(np.uint8(normalized[0]))
plt.suptitle('Normalized Image')


cropped_output = K.function([crop_layer.input],[crop_layer.output])
cropped = cropped_output([normalized])[0]
plt.figure()
#plt.imshow(np.uint8(cropped[0]))
plt.imshow(cropped[0])
plt.suptitle('Cropped Image')


# redefine model to output right after each hidden layer
ixs = [2, 3, 4, 5, 6]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# get feature map for first hidden layer
feature_maps = model.predict(np.expand_dims(image,axis=0))
# plot the output from each block

for fmap in feature_maps:
    plt.figure()
    batch,height,width,slices = fmap.shape
    ix = 1
    for i in range(slices):
        # specify subplot and turn of axis
        ax = plt.subplot(slices/4, 4, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
        ix += 1
    plt.tight_layout()
plt.show()