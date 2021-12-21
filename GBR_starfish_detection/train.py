import tensorflow.keras as keras
import tensorflow as tf
import logging
import wandb
from wandb.keras import WandbCallback
#from evaluation import evaluate



def train(config, model, ds_train, ds_val): 
    if config.optimizer=="sgd":
        opt = keras.optimizers.SGD(learning_rate=config.learning_rate)#, momentum=config.momentum)
    elif config.optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=config.learning_rate)
    else:
        print("no optimizer specified!!!")
    
    
    loss = "mean_squared_error"
    metrics = []

    model.compile(optimizer=opt, 
                    loss = loss,
                    metrics = metrics)

    model.fit(ds_train,  
                batch_size=config.batch_size,
                epochs=config.epochs,
                verbose=2,
                validation_data=ds_val,
                callbacks=[WandbCallback()])


if __name__ == "__main__":
    pass