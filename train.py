import logging
import wandb


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val):
        pass

    
    def train(model, ds_train, ds_val):

        model.compile(optimizer='adam', 
                        loss = 'sparse_categorical_crossentropy',
                        metrics = ['accuracy'])


        template = "Step: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test accuracy: {}"
        logging.info(template.format(step, 
                                    self.train_loss.result(),
                                    self.train_accuracy.result()*100,
                                    self.test_loss.result(),
                                    self.test_accuracy.result()*100))

        wandb.log({"step": step, "train_loss": self.train_loss.result(), "train_acc": self.train_accuracy.result()*100,
                    "test_loss:": self.test_loss.result(), "test_acc": self.test_accuracy.result()*100 })


        model.fit(ds_train, 
                    epochs=2, 
                    verbose=2,
                    validation_data=ds_val,
                    validation_steps=1)
