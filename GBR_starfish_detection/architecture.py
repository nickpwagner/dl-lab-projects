import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import numpy as np


def transfer_model(config): 
    # select transfer model type
    if config.architecture == "vgg16":
        base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=config.cnn_input_shape)
        preprocess_input = keras.applications.vgg16.preprocess_input
    elif config.architecture == "resnet50":
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=config.cnn_input_shape)
        preprocess_input = keras.applications.resnet50.preprocess_input
    else:
        print(f"{config.architecture} model not defined!")


    base_model.trainable = False
    inputs = keras.layers.Input(shape=config.cnn_input_shape, dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = preprocess_input(x)
    x = base_model(x)
    # one output for classification (objectness) and four for regression (bounding box coordinates)
    objectness = keras.layers.Conv2D(1, 1, 1, activation="sigmoid")(x)
    bbox = keras.layers.Conv2D(4, 1, 1, activation="linear")(x)

    # merge both output layers to a total of 5 (objectness, center_x, center_y, width, height)
    outputs = keras.layers.concatenate([objectness, bbox])

    return keras.Model(inputs=inputs, outputs=outputs, name="yolo")


if __name__ == "__main__":
    import wandb
    #from tensorflow.python.framework.ops import disable_eager_execution
    #disable_eager_execution()

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config
    
    model = transfer_model(config)
    model.summary()
