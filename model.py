import gin
import tensorflow as tf
import tensorflow.keras as keras

@gin.configurable
def vgg(input_shape, n_classes, filters, kernel, neurons, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        filters (int): number of filters
        neurons (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    inputs = keras.Input(input_shape)
    x = keras.layers.Conv2D(filters, kernel, padding='same', activation=tf.nn.relu)(inputs)
    x = keras.layers.Conv2D(filters, kernel, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D((2, 2))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(neurons, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(n_classes)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')

@gin.configurable
def resnet(input_shape, n_classes, dense0, dense1):
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape)
    print(base_model.summary())
    base_model.trainable = False
    inputs = keras.layers.Input(shape=input_shape, dtype=tf.uint8)
    x = tf.cast(inputs, tf.float32)
    x = keras.applications.resnet50.preprocess_input(x)
    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(dense0, activation=tf.nn.relu)(x)
    x = keras.layers.Dense(dense1, activation=tf.nn.relu)(x)
    outputs = keras.layers.Dense(n_classes, activation=tf.nn.softmax)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='resnet')