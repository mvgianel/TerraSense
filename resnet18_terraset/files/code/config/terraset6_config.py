#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


import os
import numpy as np

###############################################################################
# project folders
###############################################################################

def get_script_directory():
    path = os.getcwd()
    return path

# get current directory
SCRIPT_DIR = get_script_directory()

# dataset top level folder
DATASET_DIR = os.path.join(SCRIPT_DIR, "./build/dataset/terraset6")
# train, validation, test and calibration folders
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR  = os.path.join(DATASET_DIR, "test")
CALIB_DIR = os.path.join(SCRIPT_DIR, "./build/dataset/terraset6/calib")

"""
# Augmented images folder
#AUG_IMG_DIR = os.path.join(SCRIPT_DIR,'aug_img/terraset6')

# Keras model folder
KERAS_MODEL_DIR = os.path.join(SCRIPT_DIR, "build/float")

# TF checkpoints folder
CHKPT_MODEL_DIR = os.path.join(SCRIPT_DIR, "./build/tf_chkpts/terraset6")

# TensorBoard folder
TB_LOG_DIR = os.path.join(SCRIPT_DIR, "./build/tb_logs/terraset6")
"""

###############################################################################
# global variables
###############################################################################

# since we do not have validation data or access to the testing labels we need
# to take a number of images from the training data and use them instead
NUM_CLASSES      =    6
NUM_VAL_IMAGES   =  125
NUM_TEST_IMAGES  =  125
NUM_TRAIN_IMAGES = 300

#Size of images
IMAGE_WIDTH  = 32
IMAGE_HEIGHT = 32

#normalization factor to scale image 0-255 values to 0-1 #DB
NORM_FACTOR = 255.0 # could be also 256.0

# label names for the FASHION MNIST dataset
labelNames_dict = { "cobblestonebrick" : 0, "dirtground" : 1, "grass" : 2, "pavement" : 3, "sand" : 4, "stairs" : 5}
labelNames_list = ["cobblestonebrick", "dirtground", "grass", "pavement", "sand", "stairs"]



###############################################################################
# global functions
###############################################################################

'''
import cv2

_R_MEAN = 0
_G_MEAN = 0
_B_MEAN = 0

MEANS = np.array([_B_MEAN,_G_MEAN,_R_MEAN],np.dtype(np.int32))

def mean_image_subtraction(image, means):
  B, G, R = cv2.split(image)
  B = B - means[0]
  G = G - means[1]
  R = R - means[2]
  image = cv2.merge([R, G, B])
  return image
'''

def Normalize(x_test):
    x_test  = np.asarray(x_test)
    x_test = x_test.astype(np.float32)
    x_test = x_test/NORM_FACTOR
    x_test = x_test -0.5
    out_x_test = x_test *2
    return out_x_test


def ScaleTo1(x_test):
    x_test  = np.asarray(x_test)
    x_test = x_test.astype(np.float32)
    our_x_test = x_test/NORM_FACTOR
    return out_x_test
