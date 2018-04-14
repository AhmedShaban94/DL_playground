from os import environ 
environ['KERAS_BACKEND']='tensorflow' 
from keras.layers import Dense, Conv2D, Flatten, Dropout 
from keras.datasets import cifar10 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.misc import toimage 

# loading data 
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
for i in range(0, 9): 
    plt.subplot(331 + 1 +i)
    plt.imshow()