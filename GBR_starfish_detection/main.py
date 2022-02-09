import wandb
import os
import argparse
import tensorflow.keras as keras

from architecture import transfer_model
from input import load
from train import train


def main(args): 
    # read terminal config
    args_config = vars(args)
    # wandb uses the yaml file and overrides the values with the args_config
    wandb.init(project="protect_gbr", entity="stuttgartteam8", mode="online")
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config
    os.environ['WANDB_DIR'] = config.data_dir + "wandb/"
    
    # load and preprocess data set
    ds_train, _, ds_test = load(config)
    # load pre-trained transfer model and construct the rest
    detection_model = transfer_model(config)
    
    # to continue the training of an exiting model, it can be loaded from wandb
    if config.wandb_model != "New":
        print("Continue model training from wandb")
        print("Download model:")
        api = wandb.Api()
        run = api.run(config.wandb_model)
        run.file("model.h5").download(replace=True)
        print("Load model:")
        detection_model = keras.models.load_model('model.h5', compile=False) 
    
    print(detection_model.summary())
    
    # start the training
    train(config, detection_model, ds_train, ds_test)
    # save the trained model locally
    detection_model.save(os.path.join(wandb.run.dir, "model.h5"))


if __name__ == "__main__":
    # allow terminal configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_dir', type=str, help='path to the dataset and wandb logging', default=argparse.SUPPRESS)
    parser.add_argument('-m', '--wandb_model', type=str, help='name of the wandb run that stores the model', default="New")
    parser.add_argument('-d', '--dataset_slice_end', type=int, help='until which entry the csv dataframe shall be evaluated', default=argparse.SUPPRESS)
    args = parser.parse_args()
    main(args)

