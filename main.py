import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import gin
import matplotlib.pyplot as plt

from input import load
from model import vgg_like
from train import train

# required for gin functionality
gin.parse_config_files_and_bindings(['config.gin'], [])

ds_train, ds_val, ds_test = load()

for image,y in ds_train:
    print(image.shape, y)
    plt.imshow(image[0])
    plt.show()

    break

model = vgg_like(input_shape=(256, 256, 3), n_classes=5, filters=(32), kernel=(3, 3), neurons=256, dropout_rate=0.5)
model.summary()
# keras.utils.plot_model(model, show_shapes=True)

train(model, ds_train, ds_val)