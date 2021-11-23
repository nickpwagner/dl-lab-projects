import tensorflow.keras as keras
import logging
import wandb
from wandb.keras import WandbCallback
import gin


@gin.configurable
def train(model, ds_train, ds_val, optimizer, learning_rate, loss, batch_size, epochs):
    if optimizer=="sdg":
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        print("no optimizer specified!!!")
    model.compile(optimizer=opt, 
                    loss = loss,
                    metrics = ['accuracy'])

    model.fit(ds_train,  
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=ds_val,
                callbacks=[WandbCallback()])