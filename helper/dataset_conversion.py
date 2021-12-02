# file to crop and scale the images once and store the smaller dataset locally.
# image loading will be much faster
import os
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array


def crop_resize_store(input_path, output_path):
    for file in os.listdir(input_path):

        image = tf.io.read_file(input_path + file)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=560, target_height=2848, target_width=2848)
        image = tf.image.resize(image, [512, 512], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
        image = img_to_array(image)
        save_img(output_path + file, image, quality=95)


input_path = "c:/DL_Lab/IDRID_dataset_orig/images/train/"
output_path = "c:/DL_Lab/IDRID_dataset/images/train/"
crop_resize_store(input_path, output_path)

input_path = "c:/DL_Lab/IDRID_dataset_orig/images/test/"
output_path = "c:/DL_Lab/IDRID_dataset/images/test/"
crop_resize_store(input_path, output_path)
