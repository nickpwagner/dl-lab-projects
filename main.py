import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import gin
import matplotlib.pyplot as plt

import wandb
import os

from absl import flags, app  # required to pass arguments via cmdline

from input import load
from model import vgg, resnet
from train import train
from evaluation import evaluate

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('path', "C:/DL_Lab/", 'Specify the path to the dataset.')
flags.DEFINE_integer('epochs', 2, 'Specify the number of epochs to train the network.')
architecture = "resnet"

def main(argv): 
    # seeds
    # random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # required for gin functionality
    gin.parse_config_files_and_bindings(['config.gin'], [])
    

    os.environ['WANDB_DIR'] = FLAGS.path + "wandb/"
    
    default_config = {
      "learning_rate": 0.001,
      "epochs": FLAGS.epochs,
      "batch_size": 128
    }
    wandb.init(project="diabetic_retinopathy", entity="davidu", config=default_config)
    config = wandb.config

    ds_train, ds_val, ds_test = load(data_dir=FLAGS.path+"IDRID_dataset/")

    for image,y in ds_train:
        print(image.shape, y)
        plt.imshow(image[0])
        plt.show()

        break

    if architecture == "vgg":
        model = vgg()
    elif architecture == "resnet":
        model = resnet()
    else:
        print("model not supported")

    model.summary()
    # keras.utils.plot_model(model, show_shapes=True)

    if FLAGS.train:
        train(model, ds_train, ds_val) #, config.batch_size, config.epochs)

        # wandb
        # model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()])
    else:
        evaluate(model, ds_val)

if __name__ == "__main__":
    app.run(main)