#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

# Train ResNet-18 on Terraset6 data loaded into memory

# based on "Implementing ResNet-18 Using Keras" from
# https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras/notebook

# ==========================================================================================
# References
# ==========================================================================================

#https://colab.research.google.com/github/bhgtankita/MYWORK/blob/master/Grad_CAM_RESNET18_Transfer_Learning_on_CIFAR10.ipynb#scrollTo=vhO24OrY0ckv

# https://github.com/songrise/CNN_Keras

#https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/tensorflow/saving_and_serializing.ipynb#scrollTo=yKikmbdC3O_i

#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

#https://www.kaggle.com/code/parasjindal96/basic-deep-learning-tutorial-using-keras/notebook

# ==========================================================================================
# import dependencies
# ==========================================================================================

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import terraset6_config as cfg #DB
print(cfg.SCRIPT_DIR)

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

from keras.utils import np_utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,models,layers
from tensorflow.keras.utils import plot_model,  to_categorical
#from tensorflow.keras.datasets import cifar10
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
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="TF2 ResNet18 Training on Terraset6 Dataset stored as files")
    ap.add_argument("-w",  "--weights", default="build/float",      help="path to best model h5 weights file")
    #ap.add_argument("-n",  "--network", default="ResNet18_terraset6", help="input CNN")
    #ap.add_argument("-d",  "--dropout",   type=int, default=-1,     help="whether or not Dropout should be used")
    #ap.add_argument("-bn", "--BN",        type=int, default=-1,     help="whether or not BN should be used")
    ap.add_argument("-e",  "--epochs",    type=int, default=50,     help="# of epochs")
    ap.add_argument("-bs", "--batch_size",type=int, default=256,    help="size of mini-batches passed to network")
    ap.add_argument("-g",  "--gpus",      type=str, default="0",    help="choose gpu devices.")
    #ap.add_argument("-l",  "--init_lr",   type=float, default=0.01, help="initial Learning Rate")
    return ap.parse_args()

args = vars(get_arguments())
args2 = get_arguments()

for key, val in args2._get_kwargs():
    print(key+" : "+str(val))


# ==========================================================================================
# Global Variables
# ==========================================================================================

print(cfg.SCRIPT_DIR)

## Silence TensorFlow messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS  = args["weights"]
#NETWORK = args["network"]

NUM_EPOCHS = args["epochs"]     #25
#INIT_LR    = args["init_lr"]    #1e-2
BATCH_SIZE = args["batch_size"] #32

# ==========================================================================================
# prepare your data
# ==========================================================================================
print("\n[DB INFO] Loading Data for Training and Test...\n")

##ToDo add these to parameters
def load_terraset6(dataset_path, img_height=224, img_width=224, batch_size=32, validation_split=0.2, seed=123):
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

    # Function to apply random flip
    def random_flip(image, label):
        image = tf.image.random_flip_left_right(image)
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
    
    # Shuffle and split before the augmentation
    # Shuffle the original dataset
    #indices = np.arange(len(x_original))
    #np.random.shuffle(indices)
    #x_original = x_original[indices]
    #y_original = y_original[indices]

    # Split the dataset into training and validation sets
    split_index = int((1 - validation_split) * len(x_original))
    x_train, y_train = x_original[:split_index], y_original[:split_index]
    x_test, y_test = x_original[split_index:], y_original[split_index:]
    
    print("Size of training dataset before augmentation", len(x_train))
    print("Size of testing dataset before augmentation", len(x_test))
    print("Size of training + testing dataset before augmentation", (len(x_train)+len(x_test)))
    
    # Convert numpy arrays back to tf.data.Dataset
    def numpy_to_dataset(x, y):
        return tf.data.Dataset.from_tensor_slices((x,y))
        
    training_dataset = numpy_to_dataset(x_train, y_train)
    testing_dataset = numpy_to_dataset(x_test, y_test)

    # Augment each dataset by applying transformations
    augmented_training_dataset = [training_dataset]
    for preprocess_func in [random_crop, random_flip, random_brightness, random_rotation, random_zoom, random_contrast, random_saturation, random_hue]:
        augmented_training_dataset.append(training_dataset.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE))

    # Combine original and augmented datasets
        augmented_training_dataset_full = augmented_training_dataset[0]
        for aug_ds in augmented_training_dataset[1:]:
            augmented_training_dataset_full = augmented_training_dataset_full.concatenate(aug_ds)

    # Augment each dataset by applying transformations
    augmented_testing_dataset = [testing_dataset]
    for preprocess_func in [random_crop, random_flip, random_brightness, random_rotation, random_zoom, random_contrast, random_saturation, random_hue]:
        augmented_testing_dataset.append(testing_dataset.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE))

    # Combine original and augmented datasets
        augmented_testing_dataset_full = augmented_testing_dataset[0]
        for aug_ds in augmented_testing_dataset[1:]:
            augmented_testing_dataset_full = augmented_testing_dataset_full.concatenate(aug_ds) 
        

    # Convert datasets to numpy arrays
    x_train_augmented, y_train_augmented = dataset_to_numpy(augmented_training_dataset_full)
    x_test_augmented, y_test_augmented = dataset_to_numpy(augmented_testing_dataset_full)
    
    print("Size of training dataset after augmentation", len(x_train_augmented))
    print("Size of testing dataset after augmentation", len(x_test_augmented))
    print("Size of training + testing dataset after augmentation", (len(x_train_augmented)+len(x_test_augmented)))

    # Shuffle the datasets
    #indices = np.arange(len(x_train_augmented))
    #np.random.shuffle(indices)
    #x_train_augmented = x_train_augmented[indices]
    #y_train_augmented = y_train_augmented[indices]
    
    #indices = np.arange(len(x_test_augmented))
    #np.random.shuffle(indices)
    #x_test_augmented = x_test_augmented[indices]
    #y_test_augmented = y_test_augmented[indices]
    
    x_train = x_train_augmented
    y_train = y_train_augmented
    x_test = x_test_augmented
    y_test = y_test_augmented

    #import pdb; pdb.set_trace()

    return (x_train, y_train), (x_test, y_test)
# Example usage:
# (x_train, y_train), (x_test, y_test) = load_terra('path/to/terraset6')


terraset6_dataset = load_terraset6('target/terraset/terraset6')


(X_train, Y_train), (X_test, Y_test) = terraset6_dataset

#(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train.shape, X_test.shape, np.unique(Y_train).shape[0]
# one-hot encoding
n_classes = cfg.NUM_CLASSES

# Pre-processing & Normalize the data
X_train = cfg.Normalize(X_train)
X_test  = cfg.Normalize(X_test)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)

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

# data augmentation
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05,
                         height_shift_range=0.05)
aug.fit(X_train)


# ==========================================================================================
"""
ResNet-18
Reference:
[1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In
ICCV, 2015.
"""

# WARNING
# This Subclassed Model cannot be save in HDF5 format, which is what Vitis AI Quantizer requires
# see Table 17 from UG1414

"""
class ResnetBlock(Model):
    # A standard resnet block.

    def __init__(self, channels: int, down_sample=False):
        #channels: same as number of convolution kernels
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a "glorot_uniform" in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        #num_classes: number of classes in specific classification task.

        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

"""
# ==========================================================================================


# ==========================================================================================
# Get ResNet18 pre-trained model
# ==========================================================================================
print("\n[DB INFO] Get ResNet18 pretrained model...\n")

# original imagenet-based ResNet18 model
ResNet18, preprocess_input = Classifiers.get("resnet18")
#orig_model = ResNet18((224, 224, 3), weights="imagenet")
#print(orig_model.summary())


# build new model for Terraset6
#base_model = ResNet18(input_shape=(32,32,3), weights="imagenet", include_top=False)
#use model without pre-trained weights to learn from scratch on TerraSet6
base_model = ResNet18(input_shape=(224,224,3), weights=None, include_top=False)
#next to lines commented: the training would become awful
##for layer in base_model.layers:
##    layer.trainable = False
"""
dict_keys(["loss", "accuracy", "val_loss", "val_accuracy"])
X_Train Model Loss is     2.185528039932251
X_Train Model Accuracy is 0.2323250025510788
X_Test Model Loss is      2.185682535171509
X_Test Model Accuracy is  0.23180000483989716
"""
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
#model.summary()

# ==========================================================================================
# CallBack Functions
# ==========================================================================================
print("\n[DB INFO] CallBack Functions ...\n")
es = EarlyStopping(patience= 8, restore_best_weights=True, monitor="val_accuracy")


# ==========================================================================================
# Training for 50 epochs on Terraset    
# ==========================================================================================
print("\n[DB INFO] Training the Model...\n")

#use categorical_crossentropy since the label is one-hot encoded
# opt = SGD(learning_rate=0.1,momentum=0.9,decay = 1e-04) #parameters suggested by He [1]
model.compile(optimizer = "adam",loss="categorical_crossentropy", metrics=["accuracy"])


#I did not use cross validation, so the validate performance is not accurate.
STEPS = len(X_train) // BATCH_SIZE
startTime1 = datetime.now() #DB
history = model.fit(aug.flow(X_train,Y_train,batch_size = BATCH_SIZE),
                    steps_per_epoch=STEPS, batch_size = BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_data=(X_train, Y_train),callbacks=[es])
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

print("\n[DB INFO] saving HDF5 model...\n")
##model.save("resnet18_terraset6_float", save_format="tf") #TF2 saved model directory
fname1 = os.path.sep.join([WEIGHTS, "train2_resnet18_terraset6_float.h5"])
model.save(fname1) #HDF5 Keras saved model file
# once saved the model can be load with following commands #DB
#from keras.models import load_model
#print("[INFO] loading pre-trained network...") #DB
#model = load_model(fname) #DB

print("\n[DB INFO] plot model...\n")
model_filename = os.path.join(cfg.SCRIPT_DIR, "build/log/train2_float_model.png")
plot_model(model, to_file=model_filename, show_shapes=True)

# ==========================================================================================
# Prediction
# ==========================================================================================
print("\n[DB INFO] Make Predictions with Float Model on Terraset6...\n")

## Evaluation on Training Dataset
ModelLoss, ModelAccuracy = model.evaluate(X_train, Y_train)
print("X_Train Model Loss is {}".format(ModelLoss))
print("X_Train Model Accuracy is {}".format(ModelAccuracy))
"""
# expected results
X_Train Model Loss is 0.057520072907209396
X_Train Model Accuracy is 0.9808750152587891
"""

## Evaluation on Test Dataset
t_ModelLoss, t_ModelAccuracy = model.evaluate(X_test, Y_test)
print("X_Test Model Loss is {}".format(t_ModelLoss))
print("X_Test Model Accuracy is {}".format(t_ModelAccuracy))
"""
# expected results
X_Test Model Loss is 0.7710350751876831
X_Test Model Accuracy is 0.8434000015258789
"""


# make predictions on the test set
preds = model.predict(X_test)
# show a nicely formatted classification report
print(classification_report(Y_test.argmax(axis=1), preds.argmax(axis=1), target_names=cfg.labelNames_list))


# ==========================================================================================
# Training curves
# ==========================================================================================
print("\n[DB INFO] Generate Training Curves File...\n")

def plotmodelhistory(history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(history.history["accuracy"])
    axs[0].plot(history.history["val_accuracy"])
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")

    axs[0].legend(["train", "validate"], loc="upper left")
    # summarize history for loss
    axs[1].plot(history.history["loss"])
    axs[1].plot(history.history["val_loss"])
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["train", "validate"], loc="upper left")
    plt.show()

# list all data in history
print(history.history.keys())
plotmodelhistory(history)
plot_filename = os.path.join(cfg.SCRIPT_DIR, "build/log/train2_history.png")
plt.savefig(plot_filename)


# ==========================================================================================
print("\n[DB INFO] End of ResNet18 Training2 on Terraset6...\n")
