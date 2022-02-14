import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import numpy as np


def transfer_model(config): 
    """
    returns a model that uses Resnet50 as transfer learning model.
    The head can be configured/parametrized in the config.
    """
    # use Resnet50 as transfer learning model
    base_model = keras.applications.ResNet50(weights='imagenet',
                                        include_top=False,
                                        input_shape=config.cnn_input_shape)
    preprocess_input = keras.applications.resnet50.preprocess_input

    # selectable via config which layers are trainable
    if config.trainable == "full":
        for layer in base_model.layers:
            layer.trainable = True
    elif config.trainable == "last":
        for layer in base_model.layers[:-4]:
            layer.trainable = False
    elif config.trainable == "none":
        for layer in base_model.layers:
            layer.trainable = False
            
    inputs = keras.layers.Input(shape=config.cnn_input_shape, dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = preprocess_input(x)
    x = base_model(x)
    
    # output of resnet50: 14x14x2048

    if config.grid_size == 7:
        # if grid size = 7, MaxPooling is required, select a deeper architecture
        # switch between 3x3 conv and 1x1 conv as in YOLO9000 
        x = keras.layers.Conv2D(512, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Conv2D(256, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Conv2D(128, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Conv2D(128, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.MaxPool2D(pool_size=(2,2))(x)

        x = keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Conv2D(128, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Conv2D(64, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Dropout(config.dropout)(x)
        outputs = keras.layers.Conv2D(5, (1,1), strides=(1,1), padding="same", activation="linear")(x)
 
    elif config.grid_size == 14:
        # if grid size = 14, no pooling is requried
        #x = keras.layers.Conv2D(512, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        #x = keras.layers.Conv2D(256, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        #x = keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        #x = keras.layers.Conv2D(128, (1,1), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        #x = keras.layers.Conv2D(128, (3,3), strides=(1,1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        x = keras.layers.Dropout(config.dropout)(x)
        outputs = keras.layers.Conv2D(5, (1,1), strides=(1,1), padding="same", activation="linear")(x)

    else:
        print("Other grid size than 7 or 14 is not supported by architecture.py")

    return keras.Model(inputs=inputs, outputs=outputs, name="yolo")


if __name__ == "__main__":
    import wandb

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config
    
    # load the model and print the summary
    model = transfer_model(config)
    model.summary()
