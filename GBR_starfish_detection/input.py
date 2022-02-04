import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import ast
import os

class DataLoader:

    def __init__(self, config):
        self.config = config
        if not self.config.data_partial:
            self.black_list_files_train = ["00001505_skid-pad.txt", 
                        "Renningen_08_04_video_second_frame_8.txt", 
                        "Aidlingen_07_12_video_first_frame_4.txt",
                        "Aidlingen_07_12_video_first_frame_8.txt",
                        "upbracing-classes.txt",
                        "00001833_skid-pad.txt",
                        "00047.txt",
                        "Aidlingen_07_12_video_first_frame_7.txt",
                        "00001465_skid-pad.txt",
                        "Aidlingen_07_12_video_first_frame_5.txt",
                        "Aidlingen_07_12_video_first_frame_6.txt",
                        "213(2).txt",
                        "Aidlingen_07_12_video_first_frame_3.txt",
                        "00001464_skid-pad.txt"
                        ]  
                        # the labels of this samples are currupt and may not be used

        self.black_list_files_test = []


    def bbox_to_grid(config, bboxes):
        """
        read a list of labels and convert it into a grid
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
    
    def load(self):
    

    

    def convert_label_to_y(label, file):
        object_class, x_center, y_center, width_bbox, height_bbox = list(map(float, label))

        # find relevant grid cell for predicting the object
        i, j = int(y_center * config.grid_size), int(x_center * config.grid_size) # i row, j column 

        assert i < config.grid_size, "center must be within the image" + str(file.name)
        assert j < config.grid_size, "center must be within the image" + str(file.name)

        x_center_bbox = x_center * config.grid_size - j  # find center relative to cell; 0...1
        y_center_bbox = y_center * config.grid_size - i

        width_bbox *= config.grid_size  # scale box size, that 1 equals the grid size -> range of object size makes more sense
        height_bbox *= config.grid_size

        c0, c1, c2, c3, c4, c5, c6 = tf.one_hot(tf.range(7), 7)[int(object_class)] # class of the cone
        # objectness, x_center, y_center, width_bbox, height_bbox, class0, c1, c2, c3, c4, c5, c6
        y = [1.0, x_center_bbox, y_center_bbox, width_bbox, height_bbox, c0, c1, c2, c3, c4, c5, c6]
        return y, i, j


    def read_data(path, black_list_files=None):
        if black_list_files is None:
            black_list_files = []
        jpg_paths = []
        n_samples = 0
        for file in os.listdir(path):
            if file.endswith(".txt"):
                n_samples += 1
        n_samples -= len(black_list_files)

        sample_ctr = 0
        y = np.zeros((n_samples, 7, 7, 12))  # obj + bbox + classes = 1 + 4 + 7 = 12
        for file in os.listdir(path):
            if file.endswith(".txt"):
            
                if file in black_list_files:
                    continue
                #try:
                with open(os.path.join(path, file), "r") as file:
                    for label in file:
                        label = label.strip().split(" ")
                        cone, i, j = convert_label_to_y(label, file) 
                        
                        if y[sample_ctr, j, i, 0] == 0:
                            # no cone is stored at the grid cell, yet
                            y[sample_ctr, j, i, :] = cone
                        # else:
                            # slot already full- don´t write the cone
                            # print("No space to store the cone. Cone will be dismissed. File: ", file.name, i,j)
                            # think about dismissing the whole image and don´t use it for training
                            
                #except:
                #    print(str(file))
                #    continue
                #find the corresponding image path
                jpg_path = file.name.strip("txt")+"jpg"
                jpg_paths.append(jpg_path)
                sample_ctr += 1
                
        return jpg_paths, y
    ds_train = tf.data.Dataset.from_tensor_slices((jpg_paths_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((jpg_paths_test, y_test))

    def read_image(image_file, y):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image, channels=3)#, dtype=tf.float32)
        return image, y

    def resize(image, y):
        image = tf.image.resize(image, [img_height, img_width], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
        # skip rescaling, because resnet expects int8
        #image = image / 255. # rescale
        return image, y


    #ds_train = ds_train.map(read_image).map(resize_and_crop).map(augment).batch(batch_size)
    ds_train = ds_train.map(read_image).map(resize).shuffle(train_size).batch(batch_size).prefetch(2)
    ds_test = ds_test.map(read_image).map(resize).batch(batch_size)




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

    
    jpg_paths_train, y_train = read_data(config.data_dir + "train/") #, black_list_files_train)
    jpg_paths_test, y_test = read_data(config.data_dir + "test/")  

    print(y_train.shape[0], " train images found.")
    print(y_test.shape[0], " test images found.")


    """
    ds_train, _, ds_test = load(config)


    plt.figure("GBR", figsize=(10,10))
    print("show dataset")
    for images, y in ds_train:
        
        i = np.random.randint(0, config.batch_size)
        print(y[i][:,:,0])
        img = annotate_image(config, images[i], y[i], y[i])
 
        plt.imshow(img)
        plt.show()
    """
        

        

