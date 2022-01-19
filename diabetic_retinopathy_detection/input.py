import pandas as pd
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt 


def load(config):

    df_train = pd.read_csv(config.data_dir + "labels/train.csv")
    # takes val_split portion of the full df_train for training and the rest for validation. 
    y_train = df_train["Retinopathy grade"][:int(df_train.shape[0]*config.val_split)]
    img_names_train = [str(img) + ".jpg" for img in df_train["Image name"]][:int(df_train.shape[0]*config.val_split)]
    y_val = df_train["Retinopathy grade"][int(df_train.shape[0]*config.val_split):]
    img_names_val = [str(img) + ".jpg" for img in df_train["Image name"]][int(df_train.shape[0]*config.val_split):]
    df_test = pd.read_csv(config.data_dir + "labels/test.csv")
    y_test = df_test["Retinopathy grade"]
    img_names_test = [str(img) + ".jpg" for img in df_test["Image name"]]

    def create_ds(img_path, img_names, y, training=False):
        # treat unbalanced dataset with oversampling
        if config.balancing and training:

            weights = np.ones(config.n_classes) / config.n_classes
            datasets = [tf.data.Dataset.from_tensor_slices((img_names_train, y_train)).filter(lambda x, y: y==i) for i in range(config.n_classes)]
            text_ds = tf.data.experimental.sample_from_datasets(datasets, weights, stop_on_empty_dataset=True)
            #labels, counts = np.unique(y, return_counts=True)
            #weights = np.ones(len(y))
            # inverse of likelihood
            #for label in labels:
            #    weights[y == label] = np.sum(counts) / counts[label]
            
            #datasets = [tf.data.Dataset.from_tensor_slices((img_names, y)).filter(y==i) for i in range(config.n_classes)]
            #text_ds = tf.data.Dataset.sample_from_datasets(datasets, weights)
            #text_ds = tf.data.Dataset.from_tensor_slices((img_names, y, weights))
        else:  
            # creates tensors with image names and labels
            text_ds = tf.data.Dataset.from_tensor_slices((img_names, y))

        def img_name_to_image(img_names, y, weights=[]):
            X = tf.io.read_file(img_path + img_names)
            X = tf.image.decode_jpeg(X, channels=3)
            if config.mode == "multi_class":
                y = tf.one_hot(y, depth=config.n_classes)
                print("Running in multiclass classification mode!")
            elif config.mode == "binary_class":
                # y = 0,1: negative
                # y = 2,3,4: positive
                if y < 2:
                    y = 0
                else:
                    y = 1
                print("Running in binary classification mode")
            elif config.mode == "regression":
                pass
                print("Running in regression mode")
            return X, y

        def crop_and_resize_test_val(image, y):
            # if no augmentation is applied, the images need to be scaled to target resolution
            image = tf.image.central_crop(image, config.augment_crop/config.img_width)
            image = tf.image.resize(image, config.cnn_input_shape[:2], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
            return image, y

        def augment(image, y, seed):
            if config.augmentation:
                image = tf.image.stateless_random_contrast(image, 0.9, 1.1, seed=seed)
                image = tf.image.stateless_random_brightness(image, 0.1, seed=seed)
                image = tf.image.stateless_random_hue(image, 0.03, seed)
                image = tf.image.stateless_random_saturation(image, 0.9, 1.1, seed=seed)
                image = tf.image.stateless_random_flip_left_right(image, seed=seed)
                image = tf.image.stateless_random_flip_up_down(image, seed=seed)
            image = tf.image.stateless_random_crop(image, size=[config.augment_crop, config.augment_crop,3], seed=seed)
            image = tf.image.resize(image, config.cnn_input_shape[:2], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
            return image, y

        # Create a generator 
        rng = tf.random.Generator.from_seed(123, alg='philox') 
        def augment_seed(image, y):
            # random number generator specifically for stateless_random augmentation functions
            seeds = rng.make_seeds(2)[0]
            #seeds = [random.randint(0, 2**16), 42]
            image, y = augment(image, y, seeds)
            return image, y

        # img_ds takes the image names and reads the images
        if training:            
            img_ds = text_ds.map(img_name_to_image)\
                            .shuffle(len(y), reshuffle_each_iteration=True)\
                            .map(augment_seed, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(config.batch_size, drop_remainder=True)\
                            .prefetch(tf.data.AUTOTUNE)
        else:
            img_ds = text_ds.map(img_name_to_image)\
                            .shuffle(len(y), reshuffle_each_iteration=True)\
                            .map(crop_and_resize_test_val, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(config.batch_size, drop_remainder=True)\
                            .prefetch(tf.data.AUTOTUNE)
        return img_ds
    
    # e.g.  C:/DL_Lab/IDRID_dataset/   images/train/   IDRiD_001.jpg  
    ds_train = create_ds(config.data_dir + "images/train/", img_names_train, y_train, training=True)
    ds_val = create_ds(config.data_dir + "images/train/", img_names_val, y_val)
    ds_test = create_ds(config.data_dir + "images/test/", img_names_test, y_test)
    
    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    

    import matplotlib.pyplot as plt    
    import wandb

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config
    ds_train, ds_val, ds_test = load(config)

    def show_ds (ds, win_name):
        plt.figure(win_name, figsize=(10,10))
        count = 0
        ys = []
        for images, y in ds:
            for i, image in enumerate(images):
                if count < 25:
                    plt.subplot(5,5,count+1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(image/255)
                    if config.mode == "multi_class":
                        plt.title(np.argmax(y[i]))
                    else:
                        plt.title(int(y[i]))
                count += 1
                #if count >= 25:
                #    break
            ys.extend(np.argmax(y, axis=1))
            #if count >= 25:
            #    break
        print("Label distribution for " + win_name, np.unique(ys, return_counts=True))
        plt.show()
    
    show_ds(ds_train, "train_ds")
    show_ds(ds_test, "test_ds")

    # print a dataset analyis (how many samples of each class are part of the datasets)
