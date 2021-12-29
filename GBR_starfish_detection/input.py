import pandas as pd
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt 
import ast


def load(config):

    # read csv file
    df = pd.read_csv(config.data_dir + "train.csv")[config.dataset_slice_start:config.dataset_slice_end]  #[20:180]

    print(df.head())

    # read image names and convert them into the path video_id/img_id
    img_names = [f"video_{img_name.split('-')[0]}/{img_name.split('-')[1]}.jpg" for img_name in df["image_id"]]
    # read bounding box locations
    bboxes = df["annotations"]
    # translate y string from csv to 7x7x5 grid
    y = [bbox_to_grid(config, bboxes.iloc[i]) for i in range(len(bboxes))]
    len_train_ds = int(len(y)*(1-config.val_split - config.test_split))
    len_train_val_ds = int(len(y)*(1 - config.test_split))
    
    # |    train      |     val           |    test  |
    # |    60%        |     20%           |    20%   |
    #        len_train_ds       len_train_val_ds   

    text_ds_train = tf.data.Dataset.from_tensor_slices((img_names[:len_train_ds], y[:len_train_ds]))
    text_ds_val = tf.data.Dataset.from_tensor_slices((img_names[len_train_ds:len_train_val_ds], y[len_train_ds:len_train_val_ds]))
    text_ds_test = tf.data.Dataset.from_tensor_slices((img_names[len_train_val_ds:], y[len_train_val_ds:]))


    def img_name_to_image(img_names, y):
        X = tf.io.read_file(config.data_dir + "train_images/" + img_names)
        X = tf.image.decode_jpeg(X, channels=3)
        X = tf.image.resize(X, tuple(config.cnn_input_shape[:2]))
        return X, y


    # generate train data             
    ds_train = text_ds_train.map(img_name_to_image)\
                    .shuffle(int(len_train_ds/20), reshuffle_each_iteration=True)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
    # calculate length of val data set for shuffle function
    len_val_ds = int(len(y)*config.val_split)    
    # generate val data   
    ds_val = text_ds_val.map(img_name_to_image)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
    # calculate length of test data set for shuffle function
    len_test_ds = int(len(y)*config.test_split)  
    # generate test data
    ds_test = text_ds_test.map(img_name_to_image)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_val, ds_test

def convert_coord_to_bboxes(config, bboxes):
## only used to convert labels into drawable format - not required right now!!!!
    #bboxes = bboxes.numpy()
    #bboxes = bboxes.decode("UTF-8")
    bboxes = ast.literal_eval(bboxes)
    target_bboxes = []
    for bbox in bboxes:
        top_left_x = bbox["x"] / config.img_width
        top_left_y = bbox["y"] / config.img_height
        bottom_right_x = top_left_x + bbox["width"] / config.img_width 
        bottom_right_y = top_left_y + bbox["height"] / config.img_height
        target_bboxes.extend([top_left_y, top_left_x, bottom_right_y, bottom_right_x])
    return np.array(target_bboxes)

def bbox_to_grid(config, bboxes):
    y = np.zeros((config.grid_size, config.grid_size, 5))

    bboxes = ast.literal_eval(bboxes)
    # calculate grid positions for bbox centers
    for bbox in bboxes:
        # image coordinates
        x_center_abs = bbox["x"] + int(bbox["width"] / 2)
        y_center_abs = bbox["y"] + int(bbox["height"] / 2)
        # grid coordinates
        i = int(x_center_abs / config.img_width * config.grid_size) # grid cell x
        j = int(y_center_abs / config.img_height * config.grid_size) # grid cell y
        if i == config.grid_size:
            i = i-1
        if j == config.grid_size:
            j = j-1
        
        cell_size_x_abs = config.img_width/config.grid_size
        cell_size_y_abs = config.img_height/config.grid_size
        # center coordinates relative to grid cell
        x_center_rel = (x_center_abs - (i+0.5) * cell_size_x_abs ) / cell_size_x_abs
        y_center_rel = (y_center_abs - (j+0.5) * cell_size_y_abs ) / cell_size_y_abs
        # width/height relative to image
        width_abs = bbox["width"] / config.img_width
        height_abs = bbox["height"] / config.img_height
        y[i,j] = [1, x_center_rel, y_center_rel, width_abs, height_abs]
    return y

# calculate bounding box coordinates relative to its grid cell center
def grid_to_bboxes(config, grid, color="white"):
    bboxes = []
    colors = []

    for i in range(config.grid_size):
        for j in range(config.grid_size):
            objectness, x_center_rel, y_center_rel, width_abs, height_abs = grid[i,j]

            width_rel = width_abs * config.grid_size
            height_rel = height_abs * config.grid_size

            # = center_grid_cell + bbox_center_rel - bbox_width/2
            top_left_x = (i+0.5 + x_center_rel - width_rel / 2) / config.grid_size
            top_left_y = (j+0.5 + y_center_rel - height_rel / 2) / config.grid_size
            bottom_right_x = (i+0.5 + x_center_rel + width_rel / 2) / config.grid_size
            bottom_right_y = (j+0.5 + y_center_rel + height_rel / 2) / config.grid_size

            bboxes.append([top_left_y, top_left_x, bottom_right_y, bottom_right_x])
            if color == "white":
                colors.append([objectness, objectness, objectness])
            if color == "red":
                colors.append([objectness, 0, 0])
    return bboxes, colors

def show_annotated_image(config, img, grid, grid_ground_truth):
    bboxes, colors = grid_to_bboxes(config, grid)
    #if grid_ground_truth is show_annotated_image.__defaults__[3]:
        
    bboxes_gt, colors_gt = grid_to_bboxes(config, grid_ground_truth, "red")
    
    bboxes_gt.extend(bboxes)
    colors_gt.extend(colors)
        
    bboxes = np.array(bboxes_gt)
    colors = np.array(colors_gt)
    img = tf.cast(img, dtype=tf.float32)/255.
    img = tf.image.draw_bounding_boxes(tf.expand_dims(img, axis=0), bboxes.reshape([1,-1,4]), colors, name=None), 
    plt.imshow(img[0][0])
    plt.show()

def annotate_image(config, img, grid, grid_ground_truth):
    bboxes, colors = grid_to_bboxes(config, grid)
    #if grid_ground_truth is show_annotated_image.__defaults__[3]:
    bboxes_gt, colors_gt = grid_to_bboxes(config, grid_ground_truth, "red")
    bboxes_gt.extend(bboxes)
    colors_gt.extend(colors)
        
    bboxes = np.array(bboxes_gt)
    colors = np.array(colors_gt)
    img = tf.cast(img, dtype=tf.float32)/255.
    img = tf.image.draw_bounding_boxes(tf.expand_dims(img, axis=0), bboxes.reshape([1,-1,4]), colors, name=None), 
    return img[0][0]
    
def evaluate_dataset_parameter_range(ds):
    # evaluate in which range the channels are - used to determine the activation function for the final predictions
    min_max_array = np.zeros((5,2))
    for images, y in ds:
        for i in range(config.batch_size):
            for channel in range(5):
                max = np.max(y[i][:,:, channel])
                min = np.min(y[i][:,:, channel])

                if min < min_max_array[channel, 0]:
                    min_max_array[channel, 0] = min
                if max > min_max_array[channel, 1]:
                    min_max_array[channel, 1] = max
    print(min_max_array)


if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    import wandb
    import ast

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config

    
    ds_train, ds_val, ds_test = load(config)


    plt.figure("GBR", figsize=(10,10))

    for images, y in ds_train:
        #evaluate_dataset_parameter_range(ds)
        print(y.shape)
        print(y[0][:,:,0])

        show_annotated_image(config, images[0], y[0], y[0])
        #img = tf.cast(images[0], dtype=tf.float32)/255.

        #bboxes, colors = grid_to_bboxes(config, y[0])
        #colors = np.array([[1. ,0 ,0 ]])
        #img = tf.image.draw_bounding_boxes(tf.expand_dims(img, axis=0), bboxes.reshape([1,-1,4]), colors, name=None), 
        #plt.imshow(img[0][0])
        #plt.show()
 
        

        

