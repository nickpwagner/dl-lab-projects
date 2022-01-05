import wandb
import os
import argparse
import tensorflow.keras as keras

from architecture import transfer_model
from input import load, show_annotated_image
from train import train
import os




def main(args): 
    # read terminal config
    args_config = vars(args)
    # wandb uses the yaml file and overrides the values with the args_config
    wandb.init(project="protect_gbr", entity="stuttgartteam8", mode="online")
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config
    os.environ['WANDB_DIR'] = config.data_dir + "wandb/"
    
    # load and preprocess data set
    ds_train, ds_val, ds_test = load(config)

    if config.wandb_model == "New":
        # set up model architecture
        model = transfer_model(config)
    else:
        # load model from wandb and continue training
        print("Download model:")
        api = wandb.Api()
        run = api.run(config.wandb_model)
        run.file("model.h5").download(replace=True)

        print("Load model:")
        model = keras.models.load_model('model.h5', compile=False) 
        print(model.summary())

    
    # start the training
    train(config, model, ds_train, ds_val)
    # save the trained model locally
    model.save(os.path.join(wandb.run.dir, "model.h5"))


if __name__ == "__main__":
    # allow terminal configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_dir', type=str, help='path to the dataset and wandb logging', default=argparse.SUPPRESS)
    parser.add_argument('-m', '--wandb_model', type=str, help='name of the wandb run that stores the model', default="New")
    parser.add_argument('-d', '--dataset_slice_end', type=int, help='until which entry the csv dataframe shall be evaluated', default=argparse.SUPPRESS)
    args = parser.parse_args()
    main(args)

