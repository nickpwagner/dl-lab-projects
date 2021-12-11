import tensorflow.keras as keras
import logging
import wandb
from wandb.keras import WandbCallback


def train(config, model, ds_train, ds_val): 
    if config.optimizer=="sgd":
        opt = keras.optimizers.SGD(learning_rate=config.learning_rate)
    elif config.optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    else:
        print("no optimizer specified!!!")
    
    # select the loss function that corresponds to the mode
    if config.mode == "binary_class":
        loss = "binary_crossentropy"
        metric = "accuracy"
    elif config.mode == "multi_class":
        loss = "categorical_crossentropy"
        metric = "accuracy"
    elif config.mode == "regression":
        loss = "mean_squared_error"
        metric = "mean_squared_error"

    model.compile(optimizer=opt, 
                    loss = loss,
                    metrics = [metric])

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