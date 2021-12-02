import tensorflow as tf
import tensorflow.keras as keras


def transfer_model(model_type, input_shape, n_classes, dense0, dense1):
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

    x = keras.layers.GlobalAveragePooling2D()(x)
    if dense0 > 0:
        x = keras.layers.Dense(dense0, activation=tf.nn.relu)(x)
        #x = keras.layers.Dropout(0.3) 
    if dense1 > 0:
        x = keras.layers.Dense(dense1, activation=tf.nn.relu)(x)
        #x = keras.layers.Dropout(0.3)
    outputs = keras.layers.Dense(n_classes, activation=tf.nn.softmax)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=model_type)

if __name__ == "__main__":
    model = transfer_model(model_type="vgg16", input_shape=(256, 256, 3), n_classes=5, dense0=512, dense1=64)
    model.summary()
    
