import tensorflow.keras as keras
import tensorflow as tf
import logging
import wandb
from wandb.keras import WandbCallback
import tensorflow_addons as tfa


"""
def recall_m(y_train, y_pred):
    confm = tf.math.confusion_matrix(tf.math.argmax(y_train, axis=1), tf.math.argmax(y_pred, axis=1))
    
    precision = [confm[i][i] / (tf.reduce_sum(confm, axis=1)[i]+ 1e-9) for i in range(tf.shape(confm)[0])]
    return tf.reduce_mean(precision)

def precision_m(y_train, y_pred):
    confm = tf.math.confusion_matrix(tf.math.argmax(y_train, axis=1), tf.math.argmax(y_pred, axis=1))
    recalls = []
    for i in range(tf.shape(confm)[0]):
        recalls.append(tf.cast(confm[i][i], tf.float32) / (tf.cast(tf.reduce_sum(confm, axis=0)[i], tf.float32)+ 1e-9) )
    recall = tf.reduce_mean(recalls)
    return recall

def f1_m(y_train, y_pred):
    r = recall_m(y_train, y_pred)
    p = precision_m(y_train, y_pred)

    return 2*r*p/(r+p+1e-9)
"""

def train(config, model, ds_train, ds_val): 
    if config.optimizer=="sgd":
        opt = keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)
    elif config.optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    else:
        print("no optimizer specified!!!")
    
    # select the loss function that corresponds to the mode
    if config.mode == "binary_class":
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    elif config.mode == "multi_class":
        loss = "categorical_crossentropy"
        metrics = ["accuracy", tfa.metrics.F1Score(5)]    
    elif config.mode == "regression":
        loss = "mean_squared_error"
        metrics = ["mean_squared_error"]

    model.compile(optimizer=opt, 
                    loss = loss,
                    metrics = metrics)

    # define the learning rate scheduling:
    def lr_scheduler(epoch, lr):
        # exponential decay
        # e.g. if epoch = 100 and lr_decay = 10: 0.1^0.01 = 0.977 -> the factor the lr is reduced each round
        if epoch == 0:
            return lr
        return lr * (1/config.learning_rate_decay) ** (1 / (config.epochs-1))

    learning_rate_callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0) # verbose 0: quiet, verbose 1: output

    model.fit(ds_train,  
                batch_size=config.batch_size,
                epochs=config.epochs,
                verbose=2,
                validation_data=ds_val,
                callbacks=[WandbCallback(), learning_rate_callback])


    if config.fine_tuning:
        model.trainable = True
        model.summary()

        if config.optimizer=="sgd":
            opt = keras.optimizers.SGD(learning_rate=config.fine_tuning_learning_rate)
        elif config.optimizer=="adam":
            opt = keras.optimizers.Adam(learning_rate=config.fine_tuning_learning_rate)


        model.compile(optimizer=opt, 
                    loss = loss,
                    metrics = metrics)
                    
        model.fit(ds_train,  
                batch_size=config.batch_size,
                epochs=config.fine_tuning_epochs,
                verbose=2,
                validation_data=ds_val,
                callbacks=[WandbCallback()])


if __name__ == "__main__":
    pass