import gin
import tensorflow as tf

@gin.configurable
def vgg_like(input_shape, n_classes, filters, kernel, neurons, dropout_rate):
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

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(neurons, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')
