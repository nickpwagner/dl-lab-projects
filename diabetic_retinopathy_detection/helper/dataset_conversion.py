# file to crop and scale the images once and store the smaller dataset locally.
# image loading will be much faster
import os
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
import numpy as np


def crop_resize_store(input_path, output_path):
    for file in os.listdir(input_path):

        image = tf.io.read_file(input_path + file)
        image = tf.image.decode_jpeg(image, channels=3)
        ofh = 430
        ofw = 280
        image=tf.image.pad_to_bounding_box(image, offset_height=720, offset_width=0, target_height=4288, target_width=4288)
        # pad image top and bottom to not lose any of the object when cropping due to ratio
        image = tf.image.crop_to_bounding_box(image, offset_height=ofh, offset_width=ofw, target_height=4288-ofh-430, target_width=4288-ofw-580)
        # iris radius of 300 = 600 diameter
        image = tf.image.resize(image, [600, 600], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=True)
        
        # probably better with open CV (4 /-4 weighting)
        image_blur = tfa.image.gaussian_filter2d(image, (10, 10), 5)
        image = image - 0.8*image_blur
        # mean = tfa.image.mean_filter2d(image, filter_shape=600)
        # image -= mean
        image = img_to_array(image)
        save_img(output_path + file, image, quality=95)


input_path = "c:/DL_Lab/IDRID_dataset_orig/images/train/"
output_path = "c:/DL_Lab/IDRID_dataset/images/train/"
crop_resize_store(input_path, output_path)

input_path = "c:/DL_Lab/IDRID_dataset_orig/images/test/"
output_path = "c:/DL_Lab/IDRID_dataset/images/test/"
crop_resize_store(input_path, output_path)
