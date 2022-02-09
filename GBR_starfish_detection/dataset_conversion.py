# Hint:
# reads the images to scale them and store the smaller dataset locally
# image loading will be much faster
import os
import tensorflow as tf
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array


def crop_resize_store(input_path, output_path):
    """ run this function to resize the images locally to 448x448"""
    for file in os.listdir(input_path):
        image = tf.io.read_file(input_path + file)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [448, 448], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
        save_img(output_path + file, image, quality=95)

# check all dataset folders
path_extensions = ["video_0/", "video_1/", "video_2/"] 
for path_extension in path_extensions:
    input_path = "c:/DL_Lab/GBR_dataset/train_images/" + path_extension
    output_path = "c:/DL_Lab/GBR_dataset_red_size/train_images/" + path_extension
    crop_resize_store(input_path, output_path)

