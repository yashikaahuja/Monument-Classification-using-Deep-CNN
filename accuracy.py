from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--images", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
accuracy= []
directory = list(paths.list_images(args["images"]))

for images in directory:
    
    image = cv2.imread(images)
    output = image.copy()
     
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # load the trained convolutional neural network and the label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(args["model"])
    lb = pickle.loads(open(args["labelbin"], "rb").read())
     
    # classify the input image
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    accuracy.append(proba[idx]*100)
avg=0
for i in range(0,len(accuracy)):
    avg+=accuracy[i]
avg=avg/len(accuracy)
print("The overall accuracy is ")
print(avg)