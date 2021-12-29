if __name__ == "__main__":
    import wandb
    import tensorflow as tf
    from input import load, show_annotated_image

    wandb.init(project="protect_gbr", entity="stuttgartteam8", mode="disabled") 
    config = wandb.config


    ds_train, ds_val, ds_test = load(config)
    
    print("Download model:")
    api = wandb.Api()
    run = api.run("stuttgartteam8/protect_gbr/3okjp26z")
    run.file("model.h5").download(replace=True)

    print("Load model:")
    model = tf.keras.models.load_model('model.h5')
    print(model.summary())


    for x,y in ds_train:
        y_pred = model.predict(x)
        show_annotated_image(config, x[0], y_pred[0], y[0])
        #print(f"y_true: {y[0][:,:,0]}")
        #print(f"y_pred: {y_pred[0][:,:,0]}")