import pandas as pd
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt 


def load(config):
    """
    Loads the csv file and inputs the according labels and images

    Adds augmentation to the images and outputs a train, val and test dataset

    Config parameters: 
    - balancing
    - augmentation
    - augment_crop
    
    """

    # helper functions ####################################################

    def create_text_dataset(img_names, y, training=False):
        """
        takes the img name/path and the original labels and processes them into a text dataset
        """
        if config.balancing and training:
            # apply balancing only if configured and only for the training dataset
            if config.mode == "binary_class":
                # perform binary class balancing

                weights = np.array([0.5, 0.5])

                # repeat class 0; if ds1 is empty, the whole ds is empty
                ds0 = tf.data.Dataset.from_tensor_slices((img_names_train, y_train)).filter(lambda x, y: y==0 or y==1).repeat(2)
                ds1 = tf.data.Dataset.from_tensor_slices((img_names_train, y_train)).filter(lambda x, y: y==2 or y==3 or y==4)
                
                # dataset is the equally sampling of 0s and 1s
                text_ds = tf.data.experimental.sample_from_datasets([ds0, ds1], weights, stop_on_empty_dataset=True)

            else:
                # perform multiclass balancing
                weights = np.ones(config.n_classes) / config.n_classes

                # create one dataset per class to create balanced datasets afterwards
                # repeat the datasets 5 times, to ensure oversampling. 
                # The sample providing stops, if the smallest dataset (class 1) has outputed all samples 5 times
                datasets = [tf.data.Dataset.from_tensor_slices((img_names_train, y_train)).filter(lambda x, y: y==i).repeat(5) for i in range(config.n_classes)]
                            
                text_ds = tf.data.experimental.sample_from_datasets(datasets, weights, stop_on_empty_dataset=True)
        else:  
            # creates tensors with image names and labels
            text_ds = tf.data.Dataset.from_tensor_slices((img_names, y))
        return text_ds

    def img_name_to_image(img_names, y, weights=[]):
            # converts the text dataset into the actual images and does label conversioning if required
            X = tf.io.read_file(img_names)
            X = tf.image.decode_jpeg(X, channels=3)

            # label conversioning 
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
        """
        if no augmentation is applied, the images need to be scaled to target resolution
        """
        image = tf.image.central_crop(image, config.augment_crop/config.img_width)
        image = tf.image.resize(image, config.cnn_input_shape[:2], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
        return image, y

    def augment_stateless(image, y, seed):
        """
        perform augmentation on the test data
        """
        if config.augmentation == "strong":
            image = tf.image.stateless_random_contrast(image, 0.7, 1.3, seed=seed)
            image = tf.image.stateless_random_brightness(image, 0.3, seed=seed+1) # +1 to avoid correlation caused by same seeds
            image = tf.image.stateless_random_hue(image, 0.05, seed+2)
            image = tf.image.stateless_random_saturation(image, 0.8, 1.2, seed=seed+3)
            image = tf.image.stateless_random_flip_left_right(image, seed=seed+4)
            image = tf.image.stateless_random_flip_up_down(image, seed=seed+5)  
        elif config.augmentation == "weak":
            image = tf.image.stateless_random_contrast(image, 0.9, 1.1, seed=seed)
            image = tf.image.stateless_random_brightness(image, 0.1, seed=seed+1)
            image = tf.image.stateless_random_hue(image, 0.03, seed+2)
            image = tf.image.stateless_random_saturation(image, 0.9, 1.1, seed=seed+3)
            image = tf.image.stateless_random_flip_left_right(image, seed=seed+4)
            # no up-down flip for weak augmentation 
        image = tf.image.stateless_random_crop(image, size=[config.augment_crop, config.augment_crop,3], seed=seed)
        image = tf.image.resize(image, config.cnn_input_shape[:2], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
        return image, y

    # Create a generator that generates the random seed for the augmentation
    rng = tf.random.Generator.from_seed(123, alg='philox') 
    def augment(image, y):
        """
        random number generator specifically for stateless_random augmentation functions that calls the augmentation function
        """
        seeds = rng.make_seeds(2)[0]
        image, y = augment_stateless(image, y, seeds)
        return image, y


    def create_ds(img_names, y, training=False):
        """
        Create the dataset by calling the create_text_dataset and then mapping the text_ds to an image ds
        """
        
        text_ds = create_text_dataset(img_names, y, training)
        
        # img_ds takes the image names and reads the images
        if training:            
            img_ds = text_ds.map(img_name_to_image)\
                            .shuffle(len(y), reshuffle_each_iteration=True)\
                            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(config.batch_size, drop_remainder=True)\
                            .prefetch(tf.data.AUTOTUNE)
        else:
            img_ds = text_ds.map(img_name_to_image)\
                            .shuffle(len(y), reshuffle_each_iteration=True)\
                            .map(crop_and_resize_test_val, num_parallel_calls=tf.data.AUTOTUNE)\
                            .batch(config.batch_size, drop_remainder=True)\
                            .prefetch(tf.data.AUTOTUNE)
        return img_ds

    # create datasets ####################################################

    # input the image names and corresponding labels
    df_train = pd.read_csv(config.data_dir + "labels/train.csv")
    # takes val_split portion of the full df_train for training and the rest for validation. 
    y_train = df_train["Retinopathy grade"][:int(df_train.shape[0]*config.val_split)]
    # e.g.  C:/DL_Lab/IDRID_dataset/   images/train/   IDRiD_001.jpg  
    img_names_train = [config.data_dir + "images/train/" + str(img) + ".jpg" for img in df_train["Image name"]][:int(df_train.shape[0]*config.val_split)]
    y_val = df_train["Retinopathy grade"][int(df_train.shape[0]*config.val_split):]
    img_names_val = [config.data_dir + "images/train/" + str(img) + ".jpg" for img in df_train["Image name"]][int(df_train.shape[0]*config.val_split):]
    df_test = pd.read_csv(config.data_dir + "labels/test.csv")
    y_test = df_test["Retinopathy grade"]
    img_names_test = [config.data_dir + "images/test/" + str(img) + ".jpg" for img in df_test["Image name"]]

    
    ds_train = create_ds(img_names_train, y_train, training=True)
    ds_val = create_ds(img_names_val, y_val)
    ds_test = create_ds(img_names_test, y_test)
    
    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    

    import matplotlib.pyplot as plt    
    import wandb

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config
    ds_train, ds_val, ds_test = load(config)

    def show_ds (ds, win_name):
        """
        this function shows 25 images of the passed dataset and counts and prints out the quantity of samples in the dataset
        """
        plt.figure(win_name, figsize=(10,10))
        count = 0
        ys = []
        for images, y in ds:
            for i, image in enumerate(images):
                if count < 5:
                    plt.subplot(1,5,count+1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(image/255)
                    if config.mode == "multi_class":
                        plt.title(np.argmax(y[i]))
                    else:
                        plt.title(int(y[i]))
                count += 1

            if config.mode == "binary_class":
                ys.extend(y)
            else:
                ys.extend(np.argmax(y, axis=1))

        print("Label distribution for " + win_name, np.unique(ys, return_counts=True))
        plt.show()
    
    show_ds(ds_train, "train_ds")
    show_ds(ds_val, "val_ds")
    show_ds(ds_test, "test_ds")

    # print a dataset analyis (how many samples of each class are part of the datasets)
