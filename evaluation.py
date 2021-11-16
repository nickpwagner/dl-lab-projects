import tensorflow as tf
import numpy as np


def evaluate(model, ds_generator):

    y_pred = []
    y_true = []

    for X,y in ds_generator:
        y_true.extend(y)
        y_pred.extend(np.argmax(model.predict(X), axis=1))

    confm = tf.math.confusion_matrix(y_true, y_pred)#, num_classes=5)
    print(f"Confusion Matrix: {confm}")
    print(f"Accuracy: {np.mean(np.array(y_pred) == np.array(y_true)):.3f}")

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

    print(f"Precision: {np.array(precision)}")
    print(f"Recall: {np.array(recall)}")
       
    
    

    