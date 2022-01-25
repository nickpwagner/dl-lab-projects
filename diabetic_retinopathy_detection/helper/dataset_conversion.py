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
        ofh = 430
        ofw = 280
        radius = 300
        
        image=tf.image.pad_to_bounding_box(image, offset_height=720, offset_width=0, target_height=4288, target_width=4288)
        # pad image top and bottom to not lose any of the object when cropping due to ratio
        image = tf.image.crop_to_bounding_box(image, offset_height=ofh, offset_width=ofw, target_height=4288-ofh-430, target_width=4288-ofw-580)
        # iris radius of 300 = 600 diameter

        image = tf.image.resize(image, [radius*2, radius*2], method=tf.image.ResizeMethod.BILINEAR,preserve_aspect_ratio=True)

        if graham:
            image = img_to_array(image)
            # calculate gaussian blur, kernel gets calculated based on sigma
            blur = cv.GaussianBlur(image, (0,0), radius / 30)
            # subtract local mean color
            # addWeighted(img1, alpha, img2, beta, gamma)
            # image = alpha*img1+beta*img2+gamma
            # https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
            image = cv.addWeighted(image, 4, blur, -4, 128)
            # cut out the eye by masking the image
            mask = np.zeros(image.shape)
            # circle(mask, center, radius, color, thickness, linetype, shift)
            # 0.9*radius removes boundary effects
            cv.circle(mask, (image.shape[0] // 2, image.shape[1] // 2), int(0.9*radius), (1, 1, 1), -1, 8, 0)
            mask[0:70, :, :] = 0
            mask[530:, :, :] = 0
            # apply mask and change black pixels to grey
            image = (image*mask + 128 * (1-mask)) / 255
        
        save_img(output_path + file, image, quality=95)

if __name__ == "__main__":
    # input_path = "c:/DL_Lab/IDRID_dataset_orig/images/train/"
    # output_path = "c:/DL_Lab/IDRID_dataset_graham/images/train/"
    # crop_resize_store(input_path, output_path)

    # input_path = "c:/DL_Lab/IDRID_dataset_orig/images/test/"
    # output_path = "c:/DL_Lab/IDRID_dataset_graham/images/test/"
    # crop_resize_store(input_path, output_path)

    input_path = "c:/DL_Lab/IDRID_dataset_orig/images/train/"
    output_path = "c:/DL_Lab/IDRID_dataset/images/train/"
    crop_resize_store(input_path, output_path, graham=True)

    input_path = "c:/DL_Lab/IDRID_dataset_orig/images/test/"
    output_path = "c:/DL_Lab/IDRID_dataset/images/test/"
    crop_resize_store(input_path, output_path, graham=False)
