import tensorflow.keras as keras
import tensorflow as tf
import logging
import wandb
from wandb.keras import WandbCallback
from input import grid_to_bboxes
import tensorflow_addons as tfa


def train(config, model, ds_train, ds_test): 
    if config.optimizer=="sgd":
        opt = keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum)
    elif config.optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    else:
        print("no optimizer specified!!!")
    
    
    mae = keras.losses.MeanAbsoluteError()
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    mse = keras.losses.MeanSquaredError()

    def objectness_loss(y_true, y_pred):
        # objectness is stored in channel 0

        if config.objectness_loss == "mse":
            objectness_loss = mse(y_true[..., 0], y_pred[..., 0])  # is a starfish in the grid cell
        elif config.objectness_loss == "bce": 
            objectness_loss = bce(y_true[..., 0], y_pred[..., 0])  # is a starfish in the grid cell
        elif config.objectness_loss == "weighted_bce":
            objectness_loss = tf.nn.weighted_cross_entropy_with_logits(y_true[..., 0], y_pred[..., 0], config.bce_weighting)
        else:
            objectness_loss = 0
        return config.objectness_loss_weighting * objectness_loss


    def bbox_loss(y_true, y_pred):
        # objectness is stored in channels 1-4
        mask = y_true[..., 0]
        # remove cell values from all out_channels which have objectness 0
        mask = tf.stack([mask, mask, mask, mask, mask], axis=3) 
        y_pred = tf.multiply(y_pred, mask)  # mask the channels

        if config.bbox_loss == "mse":
            bbox_loss = mse(y_true[..., 1:], y_pred[..., 1:])
        elif config.bbox_loss == "mae":
            bbox_loss = mae(y_true[..., 1:], y_pred[..., 1:]) # where is the bbox located and whats the size
        elif config.bbox_loss == "giou":
            bbox_loss = gIOU(y_true, y_pred)
        return bbox_loss

    def custom_loss(y_true, y_pred):
        return objectness_loss(y_true, y_pred) + bbox_loss(y_true, y_pred)

    def yolo_loss(y_true, y_pred):
        # todo: implement exact loss function from the paper!
        pass

    def bbox_center_size_to_bbox_min_max(y):  # required for IOU and gIOU, inputs 4 channels!
            y_out = tf.zeros((16,7,7,4)) #[y_min, x_min, y_max, x_max]
            y_min = tf.subtract(y[..., 1], tf.divide(y[..., 3], 2))  # y_min = y_center - height/2
            x_min = tf.subtract(y[..., 0], tf.divide(y[..., 2], 2)) # x_min = x_center - width/2
            y_max = tf.add(y[..., 1], tf.divide(y[..., 3], 2))  # y_max = y_center + height/2
            x_max = tf.add(y[..., 0], tf.divide(y[..., 2], 2)) # x_max = x_center + width/2
            y_out = tf.stack([y_min, x_min, y_max, x_max], axis=3)
            return y_out

    def IOU(y_true, y_pred):  # used as metric!
        # assumes that the width and height is in grid-size scale
        mask = y_true[..., 0]

        y_pred = bbox_center_size_to_bbox_min_max(y_pred[..., 1:])
        y_true = bbox_center_size_to_bbox_min_max(y_true[..., 1:])

        y_pred = tf.boolean_mask(y_pred, mask)  # requried because bboxes without a true label shall not be counted
        y_true = tf.boolean_mask(y_true, mask)

        iou = tfa.losses.GIoULoss(mode="iou")
        return iou(y_true, y_pred)

    def gIOU(y_true, y_pred):# used as metric and loss
        # assumes that the width and height is in grid-size scale
        mask = y_true[..., 0]

        y_pred = bbox_center_size_to_bbox_min_max(y_pred[..., 1:])
        y_true = bbox_center_size_to_bbox_min_max(y_true[..., 1:])

        y_pred = tf.boolean_mask(y_pred, mask)  # requried because bboxes without a true label shall not be counted
        y_true = tf.boolean_mask(y_true, mask)

        iou = tfa.losses.GIoULoss(mode="giou")
        return iou(y_true, y_pred)


    def TP_rate(y_true, y_pred):
        # check if starfish objectness is correct. 
        P = tf.cast(tf.reduce_sum(y_true[..., 0]), dtype=tf.int32)  # positive in ground truth
        if P == 0:  # if there are no trues, assume TP rate = 0
            return tf.cast(0.0, dtype=tf.float64)
        else:
            y_pred = tf.where(y_pred[..., 0] > 0.5, 1, 0)  # thesholding of y_pred-objectness
            TP = tf.reduce_sum(tf.multiply(y_pred, tf.cast(y_true[..., 0], dtype=tf.int32)))  # multiply y_pred and y_true and you get the true positives
            TP_rate = TP / P
            return TP_rate

    def TN_rate(y_true, y_pred):
        N = tf.reduce_sum(tf.where(y_true[..., 0] == 0, 1, 0))  # sum up all elements where y_true = 0
        TN = tf.reduce_sum(tf.multiply(tf.where(y_pred[..., 0] < 0.5, 1, 0), tf.where(y_true[..., 0] == 0, 1, 0)))  # multiply the pred.negativ with GT negative
        TN_rate = TN / N
        return TN_rate
    
    model.compile(optimizer=opt, 
                    loss =  custom_loss,
                    metrics = [objectness_loss, bbox_loss, TP_rate, TN_rate, IOU, gIOU])

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
                validation_data=ds_test,
                callbacks=[WandbCallback(), learning_rate_callback])


if __name__ == "__main__":
    pass