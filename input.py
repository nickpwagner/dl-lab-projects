import pandas as pd
import tensorflow as tf
import gin


@gin.configurable
def load(data_dir, val_split, img_width, img_height, batch_size):

    df_train = pd.read_csv(data_dir + "labels/train.csv")

    # takes val_split portion of the full df_train for training and the rest for validation. 
    y_train = df_train["Retinopathy grade"][:int(df_train.shape[0]*val_split)]
    img_names_train = [str(img) + ".jpg" for img in df_train["Image name"]][:int(df_train.shape[0]*val_split)]

    y_val = df_train["Retinopathy grade"][int(df_train.shape[0]*val_split):]
    img_names_val = [str(img) + ".jpg" for img in df_train["Image name"]][int(df_train.shape[0]*val_split):]

    df_test = pd.read_csv(data_dir + "labels/test.csv")
    y_test = df_test["Retinopathy grade"]
    img_names_test = [str(img) + ".jpg" for img in df_test["Image name"]]

    def create_ds(img_path, img_names, y):
        
        # creates tensors with image names and labels
        text_ds = tf.data.Dataset.from_tensor_slices((img_names, y))

        def img_name_to_image(img_names, y):
            X = tf.io.read_file(img_path + img_names)
            X = tf.image.decode_jpeg(X, channels=3)
            return X, y

        def crop_and_resize(image, y):
            # image = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=560, target_height=2848, target_width=2848)
            # image = tf.image.resize(image, [img_height, img_width], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
            image = image / 255. # rescale
            return image, y

        # img_ds takes the image names and reads the images and resizes them
        img_ds = text_ds.map(img_name_to_image).map(crop_and_resize).shuffle(len(y)).batch(batch_size).prefetch(2)
        return img_ds
    
    # e.g.  C:/DL_Lab/IDRID_dataset/   images/train/   IDRiD_001.jpg  
    ds_train = create_ds(data_dir + "images/train/", img_names_train, y_train)
    ds_val = create_ds(data_dir + "images/train/", img_names_val, y_val)
    ds_test = create_ds(data_dir + "images/test/", img_names_test, y_test)
    return ds_train, ds_val, ds_test

