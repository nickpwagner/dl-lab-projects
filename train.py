import logging
import wandb
from wandb.keras import WandbCallback
import gin


@gin.configurable
def train(model, ds_train, ds_val, batch_size, epochs):

    model.compile(optimizer='adam', 
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

    model.fit(ds_train,  
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=ds_val,
                callbacks=[WandbCallback()])