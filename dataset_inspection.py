import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gin

from input import load

gin.parse_config_files_and_bindings(['config.gin'], [])

# unpack batches to lists for easier plotting
ds_train, ds_val, ds_test = load()
X_train = []
y_train = []
for X,y in ds_train:
    y_train.extend(y.numpy())
    X_train.extend(X)
    break

# print 25 images with labels for visual inspection
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(y_train[i])
plt.show()


df_train = pd.read_csv(load.data_dir + "labels/train.csv")
print(np.unique(np.array(df_train["Retinopathy grade"]), return_counts=True))