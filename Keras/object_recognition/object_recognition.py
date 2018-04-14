import os
os.environ['KERAS_BACKEND'] = 'tensorflow' 
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential   
from keras.datasets import cifar10 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.utils import np_utils 
from keras.constraints import maxnorm 
from keras.callbacks import TensorBoard 

seed = 7 
np.random.seed(seed)  

# loading Dataset 
(X_train, y_train), (X_test, y_test) = cifar10.load_data() 
# creating grid of 3x3 images 
for i in range(9): 
    plt.subplot(330 + 1 + i)
    plt.imshow(toimage(X_train[i])) 
plt.show()

# input normalization 
X_train = X_train.astype(np.float32) / 255.0 
X_test = X_test.astype(np.float32) / 255.0 

# one hot encoding for output 
y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]  

# building model 
model = Sequential() 
model.add(Conv2D(filters=32, input_shape=(3, 32, 32),
                 data_format='channels_first',kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))

# compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model fitting  
toCallback = TensorBoard(batch_size=64, write_graph=True, write_images=True, log_dir='log') 
model.fit(X_train, y_train, batch_size=64, epochs=30, verbose=1, callbacks=[toCallback], 
          validation_data=(X_test, y_test))
scores = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
print('\nmodel accuracy : %.2f%%'% (scores[1]*100))   
model.save('./model') 
