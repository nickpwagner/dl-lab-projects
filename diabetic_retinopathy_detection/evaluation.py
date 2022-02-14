import tensorflow as tf
import numpy as np
import sklearn.metrics
import argparse


def evaluate_multiclass(config, model, ds):
    """
    this function can be used to evaluate a multiclass model on the passed dataset.

    Function returns the accuracy, precision, recall, f1, confusion matrix and QWC.
    """

    y_pred = []
    y_true = []
    # loop over the full dataset and store all predictions in the y_pred / y_true lists
    for X,y in ds:
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(model.predict(X), axis=1))
    

    confm = tf.math.confusion_matrix(y_true, y_pred)#, num_classes=5)

    # calculate precision for each class
    precision = []
    recall = []
    n_rows, n_columns = np.shape(confm)
    for i in range(n_rows):
        precision_val = confm[i][i] / np.sum(confm, axis=1)[i]
        precision.append(precision_val)  

        # calculate recall
        column_sum = np.sum(confm, axis=0)[i] # use transpose of c_matrix to calculate column sums as row sums
        if (column_sum == 0): # avoid division by zero (column sum)
            recall.append(0)
        else:
            recall_val = confm[i][i] / column_sum
            recall.append(recall_val)

    acc = np.mean(np.array(y_pred) == np.array(y_true))
    p = np.mean(precision)
    r = np.mean(recall)
    f1 = 2*r*p/(r+p)
    quadratic_weighted_kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    return acc, p, r, f1, confm, quadratic_weighted_kappa
       

def evaluate_binary(config, model, ds):
    """
    this function can be used to evaluate a binary classification model on the passed dataset.

    Function returns the accuracy, precision, recall, and f1 score.
    """

    y_pred = []
    y_true = []

    for X,y in ds:
        y_true.extend(y.numpy())
        y_pred.extend(np.round(model.predict(X))[:,0])

    # compute the binary confusion matrix
    confm = tf.math.confusion_matrix(y_true, y_pred)

    # calculate precision and recall for each class
    precision = []
    recall = []
    n_rows, n_columns = np.shape(confm)
    for i in range(n_rows):
        precision_val = confm[i][i] / np.sum(confm, axis=1)[i]
        precision.append(precision_val)  

        # calculate recall
        column_sum = np.sum(confm, axis=0)[i] # use transpose of c_matrix to calculate column sums as row sums
        if (column_sum == 0): # avoid division by zero (column sum)
            recall.append(0)
        else:
            recall_val = confm[i][i] / column_sum
            recall.append(recall_val)

    acc = np.mean(np.array(y_pred) == np.array(y_true))
    p = np.mean(precision) # precision
    r = np.mean(recall) # recall
    f1 = 2*r*p/(r+p)  # f1 score 
    return acc, p, r, f1
    
def main(args): 
    """
    Load the specified wandb run and evaluate it on the validation and test set.
    """

    import wandb
    from input import load
    import os
    import time

    # wandb uses the yaml file and overrides the values with the args_config
    wandb.init(project="diabetic_retinopathy", entity="davidu", mode="disabled") 
    wandb.config.update(args, allow_val_change=True)
    config = wandb.config


    ds_train, ds_val, ds_test = load(config)
    
    model_filename = "_".join(config.evaluate_run.split("/")) + ".h5"

    # check if weights file was already downloaded before
    if os.path.isfile(model_filename):
        print("Using model from local .h5 file")
    else:
        print("Download model from wandb")
        api = wandb.Api()
        run = api.run(config.evaluate_run)
        run.file("model.h5").download(replace=True)
        time.sleep(1)
        os.rename("model.h5", model_filename)


    model = tf.keras.models.load_model(model_filename, compile=False)
    print(model.summary())

    if config.mode == "binary_class":
        print("--- Binary - Validation Scores ---")
        acc, p, r, f1 = evaluate_binary(config, model, ds_val)
        print(f"Accuracy: {acc}")
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"f1-Score: {f1}")

        print("--- Binary - Test Scores ---")
        acc, p, r, f1 = evaluate_binary(config, model, ds_test)
        print(f"Accuracy: {acc}")
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"f1-Score: {f1}")

    if config.mode == "multi_class":
        print("--- Validation Scores ---")
        acc, p, r, f1, confm, quadratic_weighted_kappa = evaluate_multiclass(config, model, ds_val)
        print(f"Accuracy: {acc}")
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"f1-Score: {f1}")
        print(f"Confusion-Matrix: \n{confm}")
        print(f"Quadratic WeightedKappa: {quadratic_weighted_kappa}")

        print("--- Test Scores ---")
        acc, p, r, f1, confm, quadratic_weighted_kappa = evaluate_multiclass(config, model, ds_test)
        print(f"Accuracy: {acc}")
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"f1-Score: {f1}")
        print(f"Confusion-Matrix: \n{confm}")
        print(f"Quadratic WeightedKappa: {quadratic_weighted_kappa}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='if binary or multiclass classification is used', default=argparse.SUPPRESS)
    parser.add_argument('-e', '--evaluate_run', type=str, help='the wandb run that shall be evaluated', default=argparse.SUPPRESS)
    parser.add_argument('-p', '--data_dir', type=str, help='path to the dataset and wandb logging', default=argparse.SUPPRESS)
    args = parser.parse_args()
    main(args)