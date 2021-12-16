import tensorflow.keras as keras
import tensorflow as tf
import logging
import wandb
from wandb.keras import WandbCallback
import tensorflow_addons as tfa
from evaluation import evaluate


class WandbLogger(tf.keras.callbacks.Callback):
    def __init__(self, config, model, ds_train, ds_val):
        super(WandbLogger, self).__init__()
        self.config = config
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val


    def on_epoch_end(self, epoch, logs):
        train_precision, train_recall, train_f1, _ = evaluate(self.config, self.model, self.ds_train)
        val_precision, val_recall, val_f1, _ = evaluate(self.config, self.model, self.ds_val)

        wandb.log({"train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1": train_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "epoch": epoch})
        

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
        metrics = ["accuracy"]    
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
                callbacks=[WandbCallback(), WandbLogger(config, model, ds_train, ds_val), learning_rate_callback])


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
                callbacks=[WandbCallback(), WandbLogger(config, model, ds_train, ds_val)])


if __name__ == "__main__":
    pass