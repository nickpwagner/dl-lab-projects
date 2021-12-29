import tensorflow.keras as keras
import tensorflow as tf
import logging
import wandb
from wandb.keras import WandbCallback
from input import grid_to_bboxes
import tensorflow_addons as tfa


def train(config, model, ds_train, ds_val): 
    if config.optimizer=="sgd":
        opt = keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)
    elif config.optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    else:
        print("no optimizer specified!!!")
    
    
    def gIOU(y_true, y_pred):

        # assumes that the width and height is in grid-size scale. Not the case since 2021-12-29 anymore.
        mse = keras.losses.MeanSquaredError()
        mask = y_true[..., 0]
        mask = tf.stack([mask, mask, mask, mask, mask], axis=3)
        y_pred = tf.multiply(y_pred, mask)

        loss = 0
        def bbox_center_size_to_bbox_min_max(y):
            y_out = tf.zeros((16,7,7,4)) #[y_min, x_min, y_max, x_max]
            y_min = tf.subtract(y[..., 2], tf.divide(y[..., 4], 2))  # y_min = y_center - height/2
            x_min = tf.subtract(y[..., 1], tf.divide(y[..., 3], 2)) # x_min = x_center - width/2
            y_max = tf.add(y[..., 2], tf.divide(y[..., 4], 2))  # y_max = y_center + height/2
            x_max = tf.add(y[..., 1], tf.divide(y[..., 3], 2)) # x_max = x_center + width/2
            y_out = tf.stack([y_min, x_min, y_max, x_max], axis=3)
            return y_out

        y_pred = bbox_center_size_to_bbox_min_max(y_pred)
        y_true = bbox_center_size_to_bbox_min_max(y_true)
        #loss = mse(y_true, y_pred)
        loss = tfa.losses.giou_loss(y_true, y_pred)
        return loss

    def custom_MSE(y_true, y_pred):
        mse = keras.losses.MeanSquaredError()
        mask = y_true[..., 0]
        mask = tf.stack([mask, mask, mask, mask, mask], axis=3)
        y_pred = tf.multiply(y_pred, mask)
        loss = mse(y_true, y_pred)
        return loss
    
    metrics = []
    model.compile(optimizer=opt, 
                    loss = custom_MSE,
                    metrics = metrics)    

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


if __name__ == "__main__":
    pass