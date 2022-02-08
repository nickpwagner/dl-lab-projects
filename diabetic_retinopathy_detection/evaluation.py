import tensorflow as tf
import numpy as np
import sklearn.metrics


def evaluate_multiclass(config, model, ds):

    if config.mode == "regression" or config.mode == "binary_class":
        print("evaluation not supported, yet.")
        return 0


    y_pred = []
    y_true = []
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

    if config.mode == "regression" or config.mode == "binary_class":
        print("evaluation not supported, yet.")
        return 0


    y_pred = []
    y_true = []

    for X,y in ds:
        y_true.extend(y)
        y_pred.extend(np.round(model.predict(X)))

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
    return acc, p, r, f1, 
    
if __name__ == "__main__":
    import wandb
    from input import load

    wandb.init(project="diabetic_retinopathy", entity="davidu", mode="disabled") 
    config = wandb.config


    ds_train, ds_val, ds_test = load(config)
    print("Keras Version: ", tf.keras.__version__)
    print("Evaluating given model")

    # model = wandb.restore('model.h5', run_path="stuttgartteam8/diabetic_retinopathy/1zktgvft")
    print("Download model:")
    api = wandb.Api()
    run = api.run(config.evaluate_run)
    run.file("model.h5").download(replace=True)

    import h5py

    f = h5py.File('Model.h5', 'r')
    print("Model Keras Version: ", f.attrs.get('keras_version'))

    print("Load model:")

    model = tf.keras.models.load_model('model.h5')
    print(model.summary())


    print("--- Validation Scores ---")
    p, r, f1, confm, quadratic_weighted_kappa = evaluate(config, model, ds_val)
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"f1-Score: {f1}")
    print(f"Confusion-Matrix: \n{confm}")
    print(f"Quadratic WeightedKappa: {quadratic_weighted_kappa}")

    print("--- Test Scores ---")
    p, r, f1, confm, quadratic_weighted_kappa = evaluate(config, model, ds_test)
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"f1-Score: {f1}")
    print(f"Confusion-Matrix: \n{confm}")
    print(f"Quadratic WeightedKappa: {quadratic_weighted_kappa}")