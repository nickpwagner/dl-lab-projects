import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import wandb
import os
import argparse

from input import load
from model import transfer_model
from train import train
from evaluation import evaluate


def main(args): 
    # seeds
    # random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    args_config = vars(args)
    wandb.init(project="diabetic_retinopathy", entity=args_config["user"])  # wandb uses the yaml file and overrides the values with the args_config
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config

    os.environ['WANDB_DIR'] = config.data_dir + "wandb/"
    

    ds_train, ds_val, ds_test = load(config.data_dir+"IDRID_dataset/", 
                                    config.val_split,
                                    config.cnn_input_shape,
                                    config.batch_size, 
                                    config.n_classes,
                                    config.crop_cut_away)

    model = transfer_model(config.architecture, tuple(config.cnn_input_shape), config.n_classes, config.head_dense0, config.head_dense1)

    if config.train:
        train(model, ds_train, ds_val, config.optimizer, \
             config.learning_rate, config.learning_rate_decay, config.loss_function,\
             config.batch_size, config.epochs)

    else:
        evaluate(model, ds_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=bool, help='True=Train, False=Evaluate', default=True)
    parser.add_argument('-u', '--user', type=str, help='specify the entity name for the wandb logging', default="stuttgartteam8")
    parser.add_argument('-p', '--data_dir', type=str, help='path to the dataset and wandb logging', default=argparse.SUPPRESS)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train the network', default=argparse.SUPPRESS)
    args = parser.parse_args()
    main(args)