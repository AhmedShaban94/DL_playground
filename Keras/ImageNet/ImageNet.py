from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions 
import numpy as np
import cv2 as cv 
import argparse

arg = argparse.ArgumentParser() 
arg.add_argument('-i', '--image', help='path to image', required=True) 
args = vars(arg.parse_args()) 

# load image and preprocessing 
original = cv.imread(args['image'])
img = image.load_img('dog.jpg', target_size=(600, 600))
img_converted = image.img_to_array(img)
img_converted = np.expand_dims(img_converted, axis=0)
final_image = preprocess_input(img_converted)


# load the Xception network
print("[INFO] loading network...")
model = Xception(weights="imagenet")

# classify the image
print("[INFO] classifying image...")
preds = model.predict(final_image)
(inID, label) = decode_predictions(preds)[0]

# display the predictions to our screen
print("ImageNet ID: {}, Label: {}".format(inID, label))
cv.putText(original, "Label: {}".format(label), (10, 30),
	cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv.imshow("Classification", original)
cv.waitKey(0)