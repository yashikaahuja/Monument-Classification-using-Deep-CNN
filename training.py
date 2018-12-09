from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from model_script.VGGNet import SmallerVGGNet
from imutils import paths
#import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
#import csv
import os

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
	# model.add(Conv2D(128, (3,3), padding="same"))
	# model.add(Activation("relu"))
	# model.add(BatchNormalization(axis=changeDim))
	# model.add(Conv2D(128, (3,3), padding="same"))
	# model.add(Activation("relu"))
	# model.add(BatchNormalization(axis=changeDim))
	# model.add(MaxPooling2D(pool_size=(2,2)))
	# model.add(Dropout(0.25))

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

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to imput dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy plot")
args = vars(ap.parse_args())

EPOCHS = 100
initial_lr = 0.001
BS = 32
IMAGE_DIMS = (96, 96, 3)

data = []
labels = []

print ("[+] loading dataset images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:

	image = cv.imread(imagePath)
	image = cv.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)


data = np.array(data, dtype="float")/255.0
labels = np.array(labels)
#print ("[+] data matrix: {:.2f}MB".format(data.nbytes/(1024*1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, 
	shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

print ("[+] compiling model...")
model = build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=initial_lr, decay=initial_lr/EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print ("[+] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print ("[+] serializing network...")
model.save(args["model"])

print ("[+] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()
