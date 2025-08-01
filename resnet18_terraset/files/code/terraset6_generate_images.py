#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


##############################################################################################

from config import terraset6_config as cfg
import glob
import numpy as np

import os
import sys
import shutil
import cv2

from random import seed
from random import random
from random import shuffle #DB
import tensorflow as tf
#from tensorflow.keras.datasets import cifar10
import gc   #Garbage collector for cleaning deleted data from memory

SCRIPT_DIR = cfg.SCRIPT_DIR
print("This script is located in: ", SCRIPT_DIR)


##############################################################################################
# make the required folders
##############################################################################################
# dataset top level
DATASET_DIR = cfg.DATASET_DIR

# train, validation and test folders
TRAIN_DIR = cfg.TRAIN_DIR
VALID_DIR = cfg.VALID_DIR
TEST_DIR  = cfg.TEST_DIR
CALIB_DIR = os.path.join(SCRIPT_DIR, "./build/dataset/terraset6/calib")

# remove any previous data
dir_list = [DATASET_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, CALIB_DIR]
for dir in dir_list:
    if (os.path.exists(dir)):
        shutil.rmtree(dir)
    os.makedirs(dir)
    print("Directory" , dir ,  "created ")

dir_list = [TRAIN_DIR, VALID_DIR, TEST_DIR, CALIB_DIR]
for dir in dir_list:
    # create sub-directories
    labeldirs = cfg.labelNames_list
    for labldir in labeldirs:
        newdir = dir + "/" + labldir
        os.makedirs(newdir, exist_ok=True)
        print("subDirectory" , newdir ,  "created ")

IMAGE_LIST_FILE = "calib_list.txt"

# create file for list of calibration images
f = open(os.path.join(CALIB_DIR, IMAGE_LIST_FILE), 'w')
imgList = list()

#############################################################################################
# Load Terraset6 Dataset
############################################################################################
# Each image is 32x32x3ch with 8bits x 1ch

##ToDo add these to parameters
def load_terraset6(dataset_path, img_height=224, img_width=224, batch_size=32, validation_split=0.2, testing_split=0.2, seed=123):
    """
    Load, augment, and preprocess the terraset6 dataset.

    Args:
    dataset_path (str): Path to the dataset directory.
    img_height (int): Target height for the images.
    img_width (int): Target width for the images.
    batch_size (int): Batch size for loading images.
    validation_split (float): Fraction of the data to reserve for validation.
    seed (int): Random seed for reproducibility.

    Returns:
    (np.array, np.array), (np.array, np.array): Training and validation datasets in the format (x_train, y_train), (x_test, y_test).
    """
   
    # Load dataset using image_dataset_from_directory without validation split
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        label_mode='int',  # or 'categorical' if you prefer one-hot encoding
        image_size=(img_height, img_width),  # Resize images to the target size
        shuffle=True,
        seed=seed
    )

    # Function to apply random crops
    def random_crop(image, label):
        image = tf.image.resize(image, [img_height + 20, img_width + 20])
        #import pdb; pdb.set_trace()
        image = tf.image.random_crop(image, size=[img_height, img_width, 3])
        return image, label
        
    def central_crop(image, label):
        cropped = tf.image.central_crop(image, central_fraction=0.8)
        image = tf.image.resize(cropped, tf.shape(image)[0:2])
        return image, label

    # Function to apply random flip left right
    def random_flip_left_right(image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label
        
    # Function to apply random flip left right
    def random_flip_up_down(image, label):
        image = tf.image.random_flip_up_down(image)
        return image, label

    # Function to apply random brightness adjustment
    def random_brightness(image, label):
        image = tf.image.random_brightness(image, max_delta=0.2)
        return image, label

    def random_rotation(image, label):
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return image, label

    def random_zoom(image, label):
        scale = tf.random.uniform([], minval=0.8, maxval=1.0)

        new_height = tf.cast(scale * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)
        new_width = tf.cast(scale * tf.cast(tf.shape(image)[1], tf.float32), tf.int32)

        image = tf.image.resize(image, [new_height, new_width])

        image = tf.image.resize_with_crop_or_pad(image, target_height=img_height, target_width=img_width)
        return image, label

    def random_contrast(image, label):  
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image, label

    def random_saturation(image, label):
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        return image, label

    def random_hue(image, label):
        image = tf.image.random_hue(image, max_delta=0.2)
        return image, label
        
    def adjust_gamma(image, label):
        image = tf.image.adjust_gamma(image, gamma=0.8)
        return image, label
        
    def random_blur(image, label):
        image = tfa.image.gaussian_filter2d(image, sigma=1.0)
        return image, label
        
    def sharpen(image, label):
        image = tfa.image.sharpness(image, factor=2.0)
        return image, label
        
    def shift_left(image, label):
        image = tf.roll(image, shift=-10, axis=1)
        return image, label
        
    def shift_right(image, label):
        image = tf.roll(image, shift=10, axis=1)
        return image, label
        
    def shift_up(image, label):
        image = tf.roll(image, shift=-10, axis=0)
        return image, label
        
    def shift_down(image, label):
        image = tf.roll(image, shift=10, axis=0)
        return image, label

    unbatched_dataset=dataset.unbatch()
    
    # Convert dataset to numpy for splitting
    def dataset_to_numpy(dataset):
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)
        
    x_original, y_original = dataset_to_numpy(unbatched_dataset)

    # Split the dataset into training and validation/testing sets
    split_index = int((1 - (validation_split + testing_split)) * len(x_original))
    x_train, y_train = x_original[:split_index], y_original[:split_index]
    x_valtest, y_valtest = x_original[split_index:], y_original[split_index:]
    
    # Split the valtest dataset into validation and testing sets (case of validation split=testing split needs to be updated)
    split_index = int((1 - 0.5) * len(x_valtest))
    x_val, y_val = x_valtest[:split_index], y_valtest[:split_index]
    x_test, y_test = x_valtest[split_index:], y_valtest[split_index:]
    
    
    print("Size of training dataset before augmentation", len(x_train))
    print("Size of validation dataset before augmentation", len(x_val))
    print("Size of testing dataset before augmentation", len(x_test))
    print("Size of training + validating + testing dataset before augmentation", (len(x_train)+ len(x_val) +len(x_test)))
    
    # Convert numpy arrays back to tf.data.Dataset
    def numpy_to_dataset(x, y):
        return tf.data.Dataset.from_tensor_slices((x,y))
        
    training_dataset = numpy_to_dataset(x_train, y_train)
    val_dataset = numpy_to_dataset(x_val, y_val)
    testing_dataset = numpy_to_dataset(x_test, y_test)
    
    augmentation_functions = [random_crop, random_brightness, random_rotation, random_zoom, random_contrast, random_saturation, random_hue, random_flip_up_down, random_flip_left_right, adjust_gamma, random_blur, central_crop, sharpen, shift_left, shift_right, shift_up, shift_down]

    def apply_augmentation(dataset, augmentation_functions):
        augmented_dataset = [dataset]
        for aug_f in augmentation_functions:
            augmented_dataset.append(dataset.map(aug_f, num_parallel_calls=tf.data.AUTOTUNE))
            
            # Combine original and augmented datasets
            augmented_dataset_full = augmented_dataset[0]
            for aug_ds in augmented_dataset[1:]:
                augmented_dataset_full = augmented_dataset_full.concatenate(aug_ds)
        
        return augmented_dataset_full
        
    augmented_training_dataset_full = apply_augmentation(training_dataset, augmentation_functions)
    augmented_val_dataset_full = apply_augmentation(val_dataset, augmentation_functions)
    augmented_testing_dataset_full = apply_augmentation(testing_dataset, augmentation_functions)
    
    # Convert datasets to numpy arrays
    x_train_augmented, y_train_augmented = dataset_to_numpy(augmented_training_dataset_full)
    x_val_augmented, y_val_augmented = dataset_to_numpy(augmented_val_dataset_full)
    x_test_augmented, y_test_augmented = dataset_to_numpy(augmented_testing_dataset_full)
    
    
    def shuffle_dataset(x, y):
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        
    # Shuffle all datasets so the augmented images mix with the originals
    shuffle_dataset(x_train_augmented, y_train_augmented)
    shuffle_dataset(x_val_augmented, y_val_augmented)
    shuffle_dataset(x_test_augmented, y_test_augmented)
    
    
    print("Size of training dataset after augmentation", len(x_train_augmented))
    print("Size of val dataset after augmentation", len(x_val_augmented))
    print("Size of testing dataset after augmentation", len(x_test_augmented))
    print("Size of training + val + testing dataset after augmentation", (len(x_train_augmented) + len(x_test_augmented) + len(x_val_augmented)))
    
    x_train = x_train_augmented
    y_train = y_train_augmented
    x_val = x_val_augmented
    y_val = y_val_augmented
    x_test = x_test_augmented
    y_test = y_test_augmented


    return (x_train, y_train), (x_test, y_test), (x_val, y_val)
    
    
terraset6_dataset = load_terraset6('target/terraset/terraset6')

(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = terraset6_dataset


########################################################################################
# convert in RGB channels (remember that OpenCV works in BGR)
# and fill the "train", "cal" and "test" folders without any classes imbalance
########################################################################################
counter1 = np.array([0,0,0,0,0,0,0,0,0,0], dtype="uint32")

num_train = 0
for i in range (0, x_train.shape[0]):
    class_name = cfg.labelNames_list[int(y_train[i])]
    # store images in TRAIN_DIR
    filename = os.path.join(TRAIN_DIR, class_name+"/"+class_name+"_"+str(i)+ ".png")
    rgb_image = x_train[i].astype("uint8")
    R,G,B = cv2.split(rgb_image)
    bgr_image = cv2.merge([B,G,R])
    #bgr_image=rgb_image
    cv2.imwrite(filename, bgr_image)
    if (i < 1000): #copy first 1000 images into CALIB_DIR too
        filename2= os.path.join(CALIB_DIR, class_name+"/"+class_name+"_"+str(i)+".png")
        local_filename = class_name+"/"+class_name+"_"+str(i)+".png"
        cv2.imwrite(filename2, bgr_image)
        imgList.append(local_filename)
    counter1[ int(y_train[int(i)]) ] = counter1[ int(y_train[int(i)]) ] +1
    num_train = num_train+1


for i in range(0, len(imgList)):
    f.write(imgList[i]+"\n")
f.close()


for i in range (0, x_test.shape[0]):
    class_name = cfg.labelNames_list[int(y_test[i])]
    # store images in TEST_DIR
    filename3=os.path.join(TEST_DIR, class_name+"/"+class_name+'_'+str(i)+'.png')
    rgb_image = x_test[i].astype("uint8")
    R,G,B = cv2.split(rgb_image)
    bgr_image = cv2.merge([B,G,R])
    #bgr_image = rgb_image
    cv2.imwrite(filename3, bgr_image)
    counter1[ int(y_test[int(i)]) ] = counter1[ int(y_test[int(i)]) ] +1

print("classes histogram in train and test dataset: ", counter1)   #DeBuG

#collect garbage to save memory
del rgb_image
gc.collect()

##############################################################################################
# split the test images into validation and test folders
##############################################################################################
# make a list of all files currently in the test folder
imagesList = [img for img in glob.glob(TEST_DIR + "/*/*.png")]

# seed random number generator
seed(1)
# randomly shuffle the list of images
shuffle(imagesList)

NVAL   = cfg.NUM_VAL_IMAGES/cfg.NUM_CLASSES
NTEST  = cfg.NUM_TEST_IMAGES/cfg.NUM_CLASSES

counter = np.array([0,0,0,0,0,0,0,0,0,0], dtype="uint32")
num_val  = 0
num_test = 0

# move the images to their class folders inside train (50000), valid (5000), test (5000)
# we want to be sure that all the folders contain same number of images per each class
for img in imagesList:
    filename = os.path.basename(img)
    classname = filename.split("_")[0]

    # read image with OpenCV
    img_orig = cv2.imread(img)
    label = cfg.labelNames_dict[classname]

    if (counter[ label ] < NTEST): #test images
        dst_dir  = TEST_DIR
        num_test = num_test+1
    else: #if (counter[ label ] >= NTEST and counter[ label ] < (NTEST+NVAL)): # validation images
        dst_dir = VALID_DIR
        num_val = num_val+1

    counter[ label ] = counter[ label ] +1;

    out_filename = os.path.join(dst_dir, cfg.labelNames_list[label]+"/"+filename)
    os.rename(img, out_filename)

print("classes histogram in train and test dataset: ", counter)
print("num images in train folder = ", num_train)
print("num images in val folder   = ", num_val)
print("num images in pred folder  = ", num_test)
print ("\nFINISHED CREATING DATASET\n")
