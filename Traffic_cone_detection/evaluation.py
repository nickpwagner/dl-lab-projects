if __name__ == "__main__":
    """
    This program is used to evaluate the performance of a YOLO detection model.

    It downloads the via config-defaults.yaml (param: wandb_run) specified wandb run and produces a video
    based on the test data.

    Make sure you select in the correct parameters for the data_dir (where your dataset is stored) and the 
    grid_size (that was selected for training (7, 14, 20).  
    """

    import wandb
    import tensorflow as tf
    from input import DataLoader, annotate_image
    import tensorflow.keras as keras
    import cv2
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    # load the wandb config from config-defaults.yaml
    wandb.init(project="protect_gbr", entity="stuttgartteam8", mode="disabled") 
    config = wandb.config


    # load and preprocess data set
    dataLoader = DataLoader(config)
    ds_train, ds_test = dataLoader.load()
    
    # prepare the output name for the model file
    model_filename = "_".join(config.wandb_run.split("/")) + ".h5"

    # check if weights file was already downloaded before
    if os.path.isfile(model_filename):
        print("Using model from local .h5 file")
    else:
        print("Download model from wandb")
        api = wandb.Api()
        run = api.run(config.wandb_run)
        run.file("model.h5").download(replace=True)
        os.rename("model.h5", model_filename)

    # load the now downloaded model
    model = tf.keras.models.load_model(model_filename, compile=False) 
    print(model.summary())


    # annotate all test images and store them in a video

    # time prefix
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M_")
    video_name = dt_string + "_".join(config.wandb_run.split("/")) + "_" + config.eval_dataset + ".mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # args: video_name, codec, fps, size
    video = cv2.VideoWriter(video_name, fourcc, 15, (config.cnn_input_shape[0],config.cnn_input_shape[1]))

    # selcect if you want to create the video based on train or test set
    if config.eval_dataset == "test":
        ds = ds_test
    elif config.eval_dataset == "train":
        ds = ds_train

    # loop over specified dataset, make predictions and add the annotated frames to the video file
    counter = 0
    for x,y in ds:
        y_pred = model.predict(x)
        for i in range(config.batch_size):
            img = annotate_image(config, x[i], y_pred[i], y[i])
            img = img*255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img.astype('uint8'))
            counter += 1
        
        print(f"{counter} images processed.")
        
        if counter > config.eval_samples:
            break
    
    # close the video
    video.release()
    