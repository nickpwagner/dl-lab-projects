import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import numpy as np


def transfer_model(config): 
    """
    load one of the pre-trained transfer models and add additional 
    conv layers for feature learning and channel reduction
    return: model with 7x7x5 output
    """
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
    
    # select how trainable the transfer model shall be 
    if config.trainable == "full":
        for layer in base_model.layers:
            layer.trainable = True
    elif config.trainable == "last":
        for layer in base_model.layers[:-4]:
            layer.trainable = False
    elif config.trainable == "none":
        for layer in base_model.layers:
            layer.trainable = False
            
    # add additional conv layers to transfer model
    inputs = keras.layers.Input(shape=config.cnn_input_shape, dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = preprocess_input(x)
    x = base_model(x)
    
    # 14x14x2048
    x = keras.layers.Conv2D(512, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    #x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Conv2D(256, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    #x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    #x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Conv2D(128, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    #x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    #x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    # 7x7x512
    
    x = keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    #x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Conv2D(128, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    #x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Conv2D(64, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x = keras.layers.Dropout(config.dropout)(x)
    outputs = keras.layers.Conv2D(5, (1,1), strides=(1,1), padding="same", activation="linear")(x)
    # 7x7x5 
    return keras.Model(inputs=inputs, outputs=outputs, name="yolo")


if __name__ == "__main__":
    # print the model layers of the transfer_model
    import wandb

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config
    model = transfer_model(config)
    model.summary()
