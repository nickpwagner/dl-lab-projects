import wandb
import os
import argparse

from architecture import transfer_model
from input import load, show_annotated_image
from train import train
import os




def main(args): 
    # read terminal config
    args_config = vars(args)
    # wandb uses the yaml file and overrides the values with the args_config
    wandb.init(project="protect_gbr", entity="stuttgartteam8", mode="disabled")
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config
    os.environ['WANDB_DIR'] = config.data_dir + "wandb/"
    
    # load and preprocess data set
    ds_train, ds_val, ds_test = load(config)
    # set up model architecture
    model = transfer_model(config)
    # start the training
    train(config, model, ds_train, ds_val)
    for x,y in ds_test:
        y_pred = model.predict(x)
        show_annotated_image(config, x[0], y_pred[0], y[0])
        print(f"y_true: {y[0][:,:,0]}")
        print(f"y_pred: {y_pred[0][:,:,0]}")
    # save the trained model locally
    model.save("model.h5")


if __name__ == "__main__":
    # allow terminal configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_dir', type=str, help='path to the dataset and wandb logging', default=argparse.SUPPRESS)
    args = parser.parse_args()
    main(args)

