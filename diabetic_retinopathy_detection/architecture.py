import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
import numpy as np


def transfer_model(config): 
    """
    this function can be used to configure an load a transfer learning model.

    Via config file, one can deside the model type (resnet50, vgg16), the number of dense channels in the head and others
    """

    # select transfer model type including the corresponding preprocessing
    if config.architecture == "vgg16":
        base_model = keras.applications.VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=config.cnn_input_shape)
        preprocess_input = keras.applications.vgg16.preprocess_input
    elif config.architecture == "resnet50":
        base_model = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=config.cnn_input_shape)
        preprocess_input = keras.applications.resnet50.preprocess_input
    else:
        print(f"{config.architecture} model not defined!")

    # select if the transfered model weights are trainable or not
    base_model.trainable = config.base_model_trainable

    inputs = keras.layers.Input(shape=config.cnn_input_shape, dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = preprocess_input(x)
    x = base_model(x)


    # select interface layer between conv and dense
    if config.c_d_interface == "gap":
        x = keras.layers.GlobalAveragePooling2D()(x)
    elif config.c_d_interface == "flatten":
        x = keras.layers.Flatten()(x)
    else:
        print("failed to convert conv layers to dense")

    # keras selects glorot_uniform as default initializer, for ReLu we choose He_normal
    # paper: https://arxiv.org/abs/1704.08863
    if config.w_init_HeNormal:
        weight_init = keras.initializers.HeNormal()
    else:
        weight_init = 'glorot_uniform' #keras.initializers.RandomNormal(stddev=0.01)

    # select how many dense layers and how many neurons each
    # only weights get regularized, not biases (except last layer)

    # dense layer 0
    if config.dense0 > 0:
        x = keras.layers.Dense(config.dense0, activation="relu", 
                kernel_initializer = weight_init, 
                kernel_regularizer = keras.regularizers.l2(config.reg_lambda))(x)
        if config.dropout > 0:
            x = keras.layers.Dropout(config.dropout)(x)

    # dense layer 1
    if config.dense1 > 0:
        x = keras.layers.Dense(config.dense1, activation="relu", 
                kernel_initializer = weight_init, 
                kernel_regularizer = keras.regularizers.l2(config.reg_lambda))(x)
        if config.dropout > 0:
            x = keras.layers.Dropout(config.dropout)(x)

    # dense layer 2
    if config.dense2 > 0:
        dense2 = keras.layers.Dense(config.dense2, activation="relu", 
                kernel_initializer = weight_init, 
                kernel_regularizer = keras.regularizers.l2(config.reg_lambda))
        x = dense2(x)
        if config.dropout > 0:
            dropout2 = keras.layers.Dropout(config.dropout)
            x = dropout2(x)

    # output layer - different for binary classification / multiclass classification / regression
    if config.mode == "binary_class":
        outputs = keras.layers.Dense(1, activation=keras.activations.sigmoid)(x)
    elif config.mode == "multi_class":
        # initialize the bias of the last dense layer for faster convergence
        bias_init_last = keras.initializers.Constant(1/config.n_classes)
        outputs = keras.layers.Dense(config.n_classes, activation=keras.activations.softmax,
                        kernel_initializer = weight_init, 
                        bias_initializer = bias_init_last,
                        kernel_regularizer = keras.regularizers.l2(config.reg_lambda))(x)
    elif config.mode == "regression":
        bias_init_last = keras.initializers.Constant(2.0)
        outputs = keras.layers.Dense(1, activation=keras.activations.linear, 
                        bias_initializer = bias_init_last,)(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name=config.mode + "_" + config.architecture)

if __name__ == "__main__":

    # load and show the architecture currently specified in the config
    import wandb

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config
    
    model = transfer_model(config)
    model.summary()
