import os 
os.environ['KERAS_BACKEND'] = 'tesnsorflow'
from keras.datasets import mnist 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import numpy as np 


# load data 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# data preprocessing 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype(np.float32)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype(np.float32) 

# generating augmented data 
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=90)
datagen.fit(X_train)

# configure batch size and retrieve one batch of images 
os.makedirs('images') 
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9,
                                     save_to_dir='images', save_format='png', save_prefix='aug'):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('Greys'))
    plt.show() 
    break 