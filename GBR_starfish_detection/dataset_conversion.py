# file to crop and scale the images once and store the smaller dataset locally.
# image loading will be much faster
import os
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Graham Preprocessing: https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801
# step by step explanation can be found at helper/open_cv_preprocessing
def crop_resize_store(input_path, output_path, graham=True):
    for file in os.listdir(input_path):
        image = tf.io.read_file(input_path + file)
        image = tf.image.decode_jpeg(image, channels=3)
        
        image = tf.image.resize(image, [448, 448], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=False)
        
        save_img(output_path + file, image, quality=95)

path_extensions = ["video_0/", "video_1/", "video_2/"] 

for path_extension in path_extensions:
    input_path = "c:/DL_Lab/GBR_dataset/train_images/" + path_extension
    output_path = "c:/DL_Lab/GBR_dataset_red_size/train_images/" + path_extension
    crop_resize_store(input_path, output_path)

