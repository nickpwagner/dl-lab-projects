import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

from input_pipeline.preprocessing import preprocess, augment

@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        # ... to do: implement dataset loading and preprocessing, shuffle, ...

        df_train = pd.read_csv(data_dir + "/IDRID_dataset/labels/train.csv")
        y_train = df_train["Retinopathy grade"][:300]
        y_val = df_train["Retinopathy grade"][300:]
        X_path_train = [str(img) + ".jpg" for img in df_train["Image name"]][:300] # first 300 images as train data
        X_path_val = [str(img) + ".jpg" for img in df_train["Image name"]][300:] # remaining ~150 images as test data

        df_test = pd.read_csv(data_dir + "labels/test.csv")
        y_test = df_test["Retinopathy grade"]
        X_path_test = [str(img) + ".jpg" for img in df_test["Image name"]]

        def create_ds(img_path, X_path, y):
            """
                returns image and label from data set gen
                
                :img_path: path to image folder (train/test)
                :X_path: image file name
                :y: labels 
            """
            text_ds = tf.data.Dataset.from_tensor_slices((X_path, y))

            def parse_img(X_path, y):
                X = tf.io.read_file(data_dir + img_path + X_path)
                X = tf.image.decode_jpeg(X, channels=3)
                return X, y

            img_ds = text_ds.map(parse_img)
            
        ds_train = create_ds("images/train/", X_path_train, y_train)
        ds_val = create_ds("images/train/", X_path_val, y_val)
        ds_test = create_ds("images/test/", X_path_train, y_test)

        ds_info = tfds.core.DataSetInfo()

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )
        print(ds_info)
        
        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info