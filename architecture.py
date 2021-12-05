import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import regularizers
import numpy as np


def transfer_model(model_type, input_shape, n_classes, dense0, dense1, dense2, c_d_interface, dropout, dropout_prob, reg_lambda):
    seed = 42

    if model_type == "vgg16":
        base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
        preprocess_input = keras.applications.vgg16.preprocess_input
    elif model_type == "resnet50":
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape)
        preprocess_input = keras.applications.resnet50.preprocess_input
    else:
        print(f"{model_type} model not defined!")
    base_model.trainable = False
    inputs = keras.layers.Input(shape=input_shape, dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = preprocess_input(x)
    x = base_model(x)

    # select interface layer between conv and dense
    if c_d_interface == "gap":
        x = keras.layers.GlobalAveragePooling2D()(x)
    elif c_d_interface == "flatten":
        x = keras.layers.Flatten()(x)

    # keras selects glorot_uniform as default initializer, for ReLu we choose He_normal
    # paper: https://arxiv.org/abs/1704.08863
    weight_init_inner = tf.keras.initializers.HeNormal()
 
    # select how many dense layers and how many neurons each
    # only weights get regularized, not biases (except last layer)
    if dense0 > 0:
        x = Dense(dense0, activation=tf.nn.relu, 
                kernel_initializer = weight_init_inner, 
                kernel_regularizer = keras.regularizers.l2(reg_lambda))(x)
        if dropout:
            x = keras.layers.Dropout(dropout_prob)(x)
    if dense1 > 0:
        x = Dense(dense1, activation=tf.nn.relu, 
                kernel_initializer = weight_init_inner, 
                kernel_regularizer = keras.regularizers.l2(reg_lambda))(x)
        if dropout:
            x = Dropout(dropout_prob)(x)
    if dense2 > 0:
        x = Dense(dense2, activation=tf.nn.relu, 
                kernel_initializer = weight_init_inner, 
                kernel_regularizer = keras.regularizers.l2(reg_lambda))(x)
        if dropout:
            x = Dropout(dropout_prob)(x)

    # initialize the bias of the last dense layer for faster convergence
    bias_init_last = tf.keras.initializers.Constant(1/n_classes)
    outputs = Dense(n_classes, activation=tf.nn.softmax,
                    kernel_initializer = weight_init_inner, 
                    bias_initializer = bias_init_last,
                    kernel_regularizer = keras.regularizers.l2(reg_lambda))(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=model_type)

if __name__ == "__main__":
    model = transfer_model(model_type="vgg16", input_shape=(256, 256, 3), n_classes=5, dense0=512, dense1=64)
    model.summary()

