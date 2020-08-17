from keras.applications.vgg16 import VGG16,decode_predictions
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = VGG16(weights='imagenet')
images = glob(".\\images\\*.jpg")

for im in images:
    img = image.load_img(im, target_size = (224,224))
    plt.imshow(img)
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x - preprocess_input(x)
    # Perform inference on our pre-processed image
    predictions = model.predict(x)

    # Check the top 3 predictions of the model
    print('Predicted:', decode_predictions(predictions, top=3)[0])
    




