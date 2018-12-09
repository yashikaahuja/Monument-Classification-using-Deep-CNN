from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import img_to_array
#from keras.optimizers import Adam
from keras import backend as K
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.model_selection import train_test_split
#from model_script.VGGNet import SmallerVGGNet
#from imutils import paths
#import matplotlib.pyplot as plt
#import numpy as np
#import argparse
#import random
#import pickle
#import cv2 as cv
#import csv
#import os

class CNNModel:
    
    def build(width, height, depth, classes):
    
    	model = Sequential()
    	inputShape = (height, width, depth)
    	changeDim = -1
    
    	if K.image_data_format() == "channels_first":
    		inputShape = (depth, height, width)
    		changeDim = 1
    
    
    	#CONV => RELU => POOL
    	model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=changeDim))
    	model.add(MaxPooling2D(pool_size=(3,3)))
    	model.add(Dropout(0.25))
    
    	#(CONV => RELU) * 2 => POOL
    	model.add(Conv2D(64, (3,3), padding="same"))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=changeDim))
    	model.add(Conv2D(64, (3,3), padding="same"))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=changeDim))
    	model.add(MaxPooling2D(pool_size=(2,2)))
    	model.add(Dropout(0.25))
    
    	#(CONV => RELU) * 2 => POOL
    	model.add(Conv2D(128, (3,3), padding="same"))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=changeDim))
    	model.add(Conv2D(128, (3,3), padding="same"))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization(axis=changeDim))
    	model.add(MaxPooling2D(pool_size=(2,2)))
    	model.add(Dropout(0.25))
    
    	#first and only set of FC => RELU layers
    	model.add(Flatten())
    	model.add(Dense(1024))
    	model.add(Activation("relu"))
    	model.add(BatchNormalization())
    	model.add(Dropout(0.5))
    
    	#softmax classifier
    	model.add(Dense(classes))
    	model.add(Activation("softmax"))
    
    	return model

