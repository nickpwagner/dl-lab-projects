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
    text_ds_test = tf.data.Dataset.from_tensor_slices((img_names[len_test_ds:], y[len_test_ds:]))


    def img_name_to_image(img_names, y):
        X = tf.io.read_file(config.data_dir + "train_images/" + img_names)
        X = tf.image.decode_jpeg(X, channels=3)
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


    # generate train data             
    ds_train = text_ds_train.map(img_name_to_image)\
                    .shuffle(int(len_train_ds/10), reshuffle_each_iteration=True)\
                    .map(augment_seed, num_parallel_calls=tf.data.AUTOTUNE)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
     
    # generate val data   
    ds_train_not_shuffled = text_ds_train.map(img_name_to_image)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
    ##################### map augment muss raus!!
    # generate test data
    ds_test = text_ds_test.map(img_name_to_image)\
                    .map(augment_seed, num_parallel_calls=tf.data.AUTOTUNE)\
                    .batch(config.batch_size, drop_remainder=True)\
                    .prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_train_not_shuffled, ds_test


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
            if objectness >= config.bbox_confidence_threshold:
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


def annotate_image(config, img, grid, grid_ground_truth):
    bboxes, colors = grid_to_bboxes(config, grid)
    #if grid_ground_truth is show_annotated_image.__defaults__[3]:
    bboxes_gt, colors_gt = grid_to_bboxes(config, grid_ground_truth, "red")
    bboxes_gt.extend(bboxes)
    colors_gt.extend(colors)
        
    bboxes = np.array(bboxes_gt)
    colors = np.array(colors_gt)
    img = tf.cast(img, dtype=tf.float32)/255.
    img = tf.expand_dims(img, axis=0)
    if len(bboxes_gt):
        img = tf.image.draw_bounding_boxes(img, bboxes.reshape([1,-1,4]), colors, name=None) 
    
    img = img[0].numpy()

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
    for images, y in ds_test:
        
        i = np.random.randint(0, config.batch_size)
        print(y[i][:,:,0])
        img = annotate_image(config, images[i], y[i], y[i])
 
        plt.imshow(img)
        plt.show()
 
        

        

