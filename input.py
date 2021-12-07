import pandas as pd
import tensorflow as tf
import numpy as np
import random


def load(data_dir, val_split, cnn_input_shape, batch_size, n_classes, crop_cut_away):

    df_train = pd.read_csv(data_dir + "labels/train.csv")

    # takes val_split portion of the full df_train for training and the rest for validation. 
    y_train = df_train["Retinopathy grade"][:int(df_train.shape[0]*val_split)]
    img_names_train = [str(img) + ".jpg" for img in df_train["Image name"]][:int(df_train.shape[0]*val_split)]

    y_val = df_train["Retinopathy grade"][int(df_train.shape[0]*val_split):]
    img_names_val = [str(img) + ".jpg" for img in df_train["Image name"]][int(df_train.shape[0]*val_split):]

    df_test = pd.read_csv(data_dir + "labels/test.csv")
    y_test = df_test["Retinopathy grade"]
    img_names_test = [str(img) + ".jpg" for img in df_test["Image name"]]

    def create_ds(img_path, img_names, y, augment_images=False):
        
        # creates tensors with image names and labels
        text_ds = tf.data.Dataset.from_tensor_slices((img_names, y))

        def img_name_to_image(img_names, y):
            X = tf.io.read_file(img_path + img_names)
            X = tf.image.decode_jpeg(X, channels=3)
            y = tf.one_hot(y, depth=n_classes)
            return X, y

        def crop_and_resize_test_val(image, y):
            # if no augmentation is applied, the images need to be scaled to target resolution
            image = tf.image.central_crop(image, 1 - crop_cut_away)
            image = tf.image.resize(image, cnn_input_shape[:2], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
            return image, y

        def crop_and_resize_train(image, y):
            boxes = np.hstack((np.random.uniform(0, crop_cut_away, (batch_size,2)), np.random.uniform(1-crop_cut_away, 1, (batch_size,2))))
            """ boxes returns an array  - for crop_cut_awy = 0.1
            array([[0.01462394, 0.08382467, 0.98080298, 0.96733774],
            [0.09881019, 0.02431573, 0.97714296, 0.9695309 ],
            [0.06304369, 0.05165648, 0.92765865, 0.92022045],
            """
            image = tf.image.crop_and_resize(image, boxes, box_indices=np.arange(batch_size), \
                                            crop_size=tuple(cnn_input_shape[:2]), method='bilinear')
            return image, y

        def augment(image, y):
            seeds =(random.randint(0, 2**16), random.randint(0, 2**16)) #(42, 42)
            seed = random.randint(0, 2**16) #42
            #image = tf.image.stateless_random_crop(image, size=[224, 224, 3], seed=seed)
            image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seed=seeds)
            image = tf.image.stateless_random_brightness(image, 0.2, seed=seeds)
            image = tf.image.stateless_random_hue(image, 0.2, seeds)
            image = tf.image.stateless_random_saturation(image, 0.5, 1.5, seed=seeds)
            image = tf.image.random_flip_left_right(image, seed=seed)
            image = tf.image.random_flip_up_down(image, seed=seed)
            #if random.random() < 0.9: image = tf.image.rgb_to_grayscale(image)
            #image = tf.image.stateless_random_jpeg_quality(image, 0.8, 1, seed=seed)
            return image, y

        # img_ds takes the image names and reads the images
        img_ds = text_ds.map(img_name_to_image).shuffle(len(y), reshuffle_each_iteration=True)
        if augment_images:
            img_ds = img_ds.map(augment).batch(batch_size, drop_remainder=True).map(crop_and_resize_train).prefetch(tf.data.AUTOTUNE)
        else:
            img_ds = img_ds.map(crop_and_resize_test_val).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return img_ds
    
    # e.g.  C:/DL_Lab/IDRID_dataset/   images/train/   IDRiD_001.jpg  
    ds_train = create_ds(data_dir + "images/train/", img_names_train, y_train, augment_images=True)
    ds_val = create_ds(data_dir + "images/train/", img_names_val, y_val)
    ds_test = create_ds(data_dir + "images/test/", img_names_test, y_test)
    return ds_train, ds_val, ds_test

if __name__ == "__main__":
    import matplotlib.pyplot as plt    

    ds_train, ds_val, ds_test = load("C:/DL_Lab/IDRID_dataset/", 0.8, [224, 224, 3], 16, 5, 0.1)

    plt.figure(figsize=(10,10))
    count = 0
    for image,y in ds_train:
        plt.subplot(5,5,count+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image[0]/255)
        plt.title(np.argmax(y[0]))

        count += 1
        if count == 24:
            break
    plt.show()

    # for image,y in ds_train:
    #     print(image.shape)
    #     plt.imshow(image[0]/255)
    #     plt.show()

    #     break

    # for image,y in ds_test:
    #     print(image.shape)
    #     plt.imshow(image[0]/255)
    #     plt.show()

    #     break

    # todo´s
    # print several augmented and not augmented images
    # print a dataset analyis (how many samples of each class are part of the datasets)
