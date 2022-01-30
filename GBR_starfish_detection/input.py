import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import ast


def load(config):

    # read csv file
    df = pd.read_csv(config.data_dir + "train.csv")[config.dataset_slice_start:config.dataset_slice_end]  #[20:180]

    print(df.head())
    print(f"Loading {len(df)} images.")

    # read image names and convert them into the path video_id/img_id
    img_names = [f"video_{img_name.split('-')[0]}/{img_name.split('-')[1]}.jpg" for img_name in df["image_id"]]
    # read bounding box locations
    bboxes = df["annotations"]
    # translate y string from csv to 7x7x5 grid
    y = [bbox_to_grid(config, bboxes.iloc[i]) for i in range(len(bboxes))]
    len_train_ds = int(len(y)*(1 - config.test_split))
    len_test_ds = int(len(y) * config.test_split)
    
    # |    train      |         test  |
    # |    80%        |         20%   |
    #        len_train_ds          

    text_ds_train = tf.data.Dataset.from_tensor_slices((img_names[:len_train_ds], y[:len_train_ds]))
    text_ds_test = tf.data.Dataset.from_tensor_slices((img_names[len_train_ds:], y[len_train_ds:]))


    def img_name_to_image(img_names, y):
        X = tf.io.read_file(config.data_dir + "train_images/" + img_names)
        X = tf.image.decode_jpeg(X, channels=3)
        # 1280x720 to cnn_input_shape size
        X = tf.image.resize(X, tuple(config.cnn_input_shape[:2]))
        return X, y

    def augment(image, y, seed):
        image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seed=seed)
        image = tf.image.stateless_random_brightness(image, 0.1, seed=seed)
        image = tf.image.stateless_random_hue(image, 0.05, seed)
        image = tf.image.stateless_random_saturation(image, 0.9, 1.1, seed=seed)

        image_flip_lr = tf.image.stateless_random_flip_left_right(image, seed=seed)
        if tf.math.reduce_all(tf.equal(image, image_flip_lr)) == False:
            y = tf.reverse(y, axis=[0]) # row, cols, 5 -> row = 0
            y = tf.stack([y[:,:, 0], -y[:,:, 1], y[:,:, 2], y[:,:, 3], y[:,:, 4]], axis=2)
            
        image_flip_ud = tf.image.stateless_random_flip_up_down(image_flip_lr, seed=seed)
        if tf.math.reduce_all(tf.equal(image_flip_lr, image_flip_ud)) == False:
            y = tf.reverse(y, axis=[1]) # row, cols, 5 -> cols = 1
            y = tf.stack([y[:,:, 0], y[:,:, 1], -y[:,:, 2], y[:,:, 3], y[:,:, 4]], axis=2)
        return image_flip_ud, y

    # Create a generator 
    rng = tf.random.Generator.from_seed(123, alg='philox') 
    def augment_seed(image, y):
        # random number generator specifically for stateless_random augmentation functions
        seeds = rng.make_seeds(2)[0]
        #seeds = [random.randint(0, 2**16), 42]
        image, y = augment(image, y, seeds)
        return image, y


    # generate train data  - more data than 1/7 th of the train DS doesn´t fit into RAM           
    ds_train = text_ds_train.map(img_name_to_image)\
                    .shuffle(int(len_train_ds/7), reshuffle_each_iteration=True)\
                    .map(augment_seed, num_parallel_calls=tf.data.AUTOTUNE)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
     
    # generate val data   
    ds_train_not_shuffled = text_ds_train.map(img_name_to_image)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
    # generate test data
    ds_test = text_ds_test.map(img_name_to_image)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_train_not_shuffled, ds_test


def bbox_to_grid(config, bboxes):
    """
    read csv entry for bbox and transform it to matrices for x, y, width and height
    e.g. {"x": 559, "y": 213, "width":50, "height": 32} to 7x7x5 tensor (including objectness)
    return: y
    """
    y = np.zeros((config.grid_size, config.grid_size, 5))

    bboxes = ast.literal_eval(bboxes)
    # calculate grid positions for bbox centers
    for bbox in bboxes:
        # absolute coordinates for center 
        # e.g. x_abs = position + half the width 
        x_center_abs = bbox["x"] + int(bbox["width"] / 2)
        y_center_abs = bbox["y"] + int(bbox["height"] / 2)
        # to which grid cell do those coordinates belong
        # e.g. x_center_abs: 584 to cell 3
        i = int(x_center_abs / config.img_width * config.grid_size) # grid cell x
        j = int(y_center_abs / config.img_height * config.grid_size) # grid cell y
        if i == config.grid_size:
            i = i-1
        if j == config.grid_size:
            j = j-1
        # absolute width/height per cell
        cell_size_x_abs = config.img_width/config.grid_size
        cell_size_y_abs = config.img_height/config.grid_size
        # center coordinates relative to grid cell
        # e.g. x_center_rel = (584 - 3.5 * 183) / 183 = -0.3 (left of center of cell 3)
        x_center_rel = (x_center_abs - (i+0.5) * cell_size_x_abs ) / cell_size_x_abs
        y_center_rel = (y_center_abs - (j+0.5) * cell_size_y_abs ) / cell_size_y_abs
        # width/height coordinates relative to grid cell
        # e.g. width_rel = 50 / 1280 * 7 = 0.27 grid cells
        width_rel = bbox["width"] / config.img_width * config.grid_size
        height_rel = bbox["height"] / config.img_height * config.grid_size
        # write [objectness, center_x, center_y, width, height] into bbox grid cell
        y[i,j] = [1, x_center_rel, y_center_rel, width_rel, height_rel]
    return y


def grid_to_bboxes(config, grid, color="white"):
    """
    check each cell for a bounding box in relative format [objectness, center_x, center_y, width, height]
    and transform it to absolute coordinates [x_topleft, y_topleft, x_bottomright, y_bottomright]    
    return: list of bboxes for e.g. annotation in image (normalized to [0,1]) and their color
    """
    bboxes = []
    colors = []
    # iterate through grid cells
    for i in range(config.grid_size):
        for j in range(config.grid_size):
            # relative coordinates
            objectness, x_center_rel, y_center_rel, width_rel, height_rel = grid[i,j]
            # non-maxima suppression with configurable threshold
            if objectness >= config.bbox_confidence_threshold:
                # add grid center position relative center to get grid position
                # and subtract half the bbox width to get the corner coordinate (+normalized)
                top_left_x = (i+0.5 + x_center_rel - width_rel / 2) / config.grid_size
                top_left_y = (j+0.5 + y_center_rel - height_rel / 2) / config.grid_size
                bottom_right_x = (i+0.5 + x_center_rel + width_rel / 2) / config.grid_size
                bottom_right_y = (j+0.5 + y_center_rel + height_rel / 2) / config.grid_size
                # fill list with bboxes
                bboxes.append([top_left_y, top_left_x, bottom_right_y, bottom_right_x])
                if color == "white":
                    colors.append([objectness, objectness, objectness])
                if color == "red":
                    colors.append([objectness, 0, 0])
    return bboxes, colors


def annotate_image(config, img, y_pred, y_true):
    """
    take input image and draw predicted and true bounding boxes
    returns: image with bboxes
    """
    # change from y_pred to [x_topleft, y_topleft, x_bottomright, y_bottomright] format
    bboxes_pred, colors_pred = grid_to_bboxes(config, y_pred)
    # change from y_true to [x_topleft, y_topleft, x_bottomright, y_bottomright] format
    bboxes_true, colors_true = grid_to_bboxes(config, y_true, "red")
    # put all boxes in one list and their colors in another
    bboxes_true.extend(bboxes_pred)
    colors_true.extend(colors_pred)
    bboxes_pred = np.array(bboxes_true)
    colors_pred = np.array(colors_true)
    # prepare image for draw function
    img = tf.cast(img, dtype=tf.float32)/255.
    img = tf.expand_dims(img, axis=0)
    # if there are boxes to be drawn in the image
    if len(bboxes_true):
        # function expects a batch size so expand dims by 1
        img = tf.image.draw_bounding_boxes(img, bboxes_pred.reshape([1,-1,4]), colors_pred, name=None) 
    img = img[0].numpy()
    # clip image to [0,1]
    img[img>1.0] = 1.0
    img[img<0.0] = 0.0
    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt    
    import wandb
    import ast

    wandb.init(project="test", entity="team8", mode="disabled") 
    config = wandb.config

    print("Load dataset")
    ds_train, _, ds_test = load(config)


    plt.figure("GBR", figsize=(10,10))
    print("show dataset")
    for images, y in ds_train:
        
        i = np.random.randint(0, config.batch_size)
        print(y[i][:,:,0])
        img = annotate_image(config, images[i], y[i], y[i])
 
        plt.imshow(img)
        plt.show()
 
        

        

