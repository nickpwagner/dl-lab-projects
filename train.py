import tensorflow.keras as keras
import logging
import wandb
from wandb.keras import WandbCallback


def train(model, ds_train, ds_val, optimizer, learning_rate, learning_rate_decay, loss, batch_size, epochs):
    if optimizer=="sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print("no optimizer specified!!!")

    model.compile(optimizer=opt, 
                    loss = loss,
                    metrics = ['accuracy'])

    # define the learning rate scheduling:
    def lr_scheduler(epoch, lr):
        # exponential decay
        # e.g. if epoch = 100 and lr_decay = 10: 0.1^0.01 = 0.977 -> the factor the lr is reduced each round
        if epoch == 0:
            return lr
        return lr * (1/learning_rate_decay) ** (1 / (epochs-1))

    learning_rate_callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1) # verbose 0: quiet, verbose 1: output

    model.fit(ds_train,  
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=ds_val,
                callbacks=[WandbCallback(), learning_rate_callback])