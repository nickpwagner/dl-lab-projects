import wandb
from wandb.keras import WandbCallback

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import gin
import matplotlib.pyplot as plt

from input import load
from model import vgg_like
from train import train
from evaluation import evaluate

# required for gin functionality
gin.parse_config_files_and_bindings(['config.gin'], [])

# init wandb with the correct user account
wandb.init(project="my-test-project", entity="davidu")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
 

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

# wandb
# model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()])



#evaluate(model, ds_val)

