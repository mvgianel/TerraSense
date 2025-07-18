#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# ==========================================================================================
# import dependencies
# ==========================================================================================


from config import terraset6_config as cfg #DB

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.utils import np_utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,models,layers
from tensorflow.keras.utils import plot_model
#rom tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD

from classification_models.keras import Classifiers



# ==========================================================================================
# Get Input Arguments
# ==========================================================================================
def get_arguments():
    """Parse all the arguments.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Vitis AI TF2 Quantization of ResNet18 trained on Terraset6")

    # model config
    parser.add_argument("--float_file", type=str, default="./build/float/train2_resnet18_terraset6.h5",
                        help="h5 floating point file full path name")
    # others
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu devices.")
    # quantization config
    parser.add_argument("--quant_file", type=str, default="./build/quantized/q_train2_resnet18_terraset6.h5",
                        help="quantized model file full path ename ")

    return parser.parse_args()


#def main():
args = get_arguments()

# ==========================================================================================
# Global Variables
# ==========================================================================================
print(cfg.SCRIPT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

FLOAT_HDF5_FILE = os.path.join(cfg.SCRIPT_DIR,  args.float_file)
QUANT_HDF5_FILE = os.path.join(cfg.SCRIPT_DIR,  args.quant_file)


# ==========================================================================================
# prepare your data
# ==========================================================================================
print("\n[DB INFO] Preparing Data ...\n")

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

X_train.shape, X_test.shape, X_val.shape, np.unique(Y_train).shape[0]
# one-hot encoding
n_classes = cfg.NUM_CLASSES

# Pre-processing & Normalize the data
X_train = cfg.Normalize(X_train)
X_test  = cfg.Normalize(X_test)
X_val  = cfg.Normalize(X_val)

#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)
#import pdb; pdb.set_trace()
#need to reshape in order to apply encoding
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
Y_val = Y_val.reshape(-1,1)

#import pdb; pdb.set_trace()
encoder = OneHotEncoder()
encoder.fit(Y_train)
Y_train = encoder.transform(Y_train).toarray()
Y_test = encoder.transform(Y_test).toarray()
Y_val =  encoder.transform(Y_val).toarray()


# ==========================================================================================
# Get the trained floating point model
# ==========================================================================================

model = keras.models.load_model(FLOAT_HDF5_FILE)

# ==========================================================================================
# Prediction
# ==========================================================================================
print("\n[DB INFO] Make Predictions with Float Model...\n")

## Evaluation on Training Dataset
ModelLoss, ModelAccuracy = model.evaluate(X_train, Y_train)
print("X_Train Model Loss     is {}".format(ModelLoss))
print("X_Train Model Accuracy is {}".format(ModelAccuracy))

## Evaluation on Test Dataset
t_ModelLoss, t_ModelAccuracy = model.evaluate(X_test, Y_test)
print("X_Test Model Loss     is {}".format(t_ModelLoss))
print("X_Test Model Accuracy is {}".format(t_ModelAccuracy))


# ==========================================================================================
# Vitis AI Quantization
# ==========================================================================================
print("\n[DB INFO] Vitis AI Quantization...\n")

from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)
q_model = quantizer.quantize_model(calib_dataset=X_train[0:100])
#q_model.save(QUANT_HDF5_FILE)

print("\n[DB INFO] Evaluation of Quantized Model...\n")
with vitis_quantize.quantize_scope():
    #q_model = tf.keras.models.load_model("./quantized.h5", custom_objects=custom_objects)
    #q_model = tf.keras.models.load_model(QUANT_HDF5_FILE)
    q_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    q_eval_results = q_model.evaluate(X_test, Y_test)
    print("\n***************** Summary *****************")
    print("X_Test Quantized model accuracy: ", q_eval_results[1])

print("\n[DB INFO] Saving Quantized Model...\n")
q_model.save(QUANT_HDF5_FILE)
loaded_model = keras.models.load_model(QUANT_HDF5_FILE)
#eval_results = loaded_model.evaluate(X_test, Y_test)
#print("\n***************** Summary *****************")
#print("X_Test Quantized model accuracy: ", eval_results[1])
eval_results = loaded_model.evaluate(X_train, Y_train)
print("\n***************** Summary *****************")
print("X_Train Quantized model accuracy: ", eval_results[1])
