from os import environ 
environ['KERAS_BACKEND'] =  'tensorflow'  
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten   
from keras.datasets import mnist 
import numpy as np 
import matplotlib.pyplot as plt 
from keras.utils import np_utils 
from keras.callbacks import TensorBoard   
seed = 7 
np.random.seed(seed)

# loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

# falttening images 
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) 
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

#normalizing the input images from 0-255 (grey-scale) to 0-1 
X_train = X_train / 255 
X_test = X_test / 255   

# one hot encoding for outputs 
y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 

# importnant parameters 
num_classes = y_train.shape[1] 
samples = X_train.shape[0] 
inputs = X_train.shape[1] 

# baseline model with multi-layer perceptron 
def baseline_model(): 
    model = Sequential()
    model.add(Dense(units=inputs, input_dim=inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax', kernel_initializer='normal'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model 

model = baseline_model() 
tensorboard = TensorBoard(log_dir="./graph_NN", histogram_freq=0, write_graph=True, write_images=True)
tensorboard.set_model(model) 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1, batch_size=16, callbacks=[tensorboard])

# model evaluation 
scores = model.evaluate(X_test, y_test, verbose=1) 
print(' \nBaseline error %.2f%%' % (100 - scores[1]*100))



# Building convolutional Neural Network model 

# loading data
(X_train ,y_train), (X_test, y_test) = mnist.load_data() 

# reshaping data to [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype(np.float32)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype(np.float32)

# normalizing input images 
X_train = X_train / 255 
X_test = X_test / 255  

# one hot encoding for outputs 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test) 

# importnant parameters 
num_classes = y_train.shape[1] 

def CNN_model():
    model = Sequential()
    model.add(Conv2D(filters=30, input_shape=(28, 28, 1), data_format='channels_last',
                     activation='relu', kernel_initializer='normal', kernel_size=(5, 5)))
    #note: you can write (1, 28, 28) with data_format='channels_first' 
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=15, kernel_size=(3, 3),
                     kernel_initializer='normal', data_format='channels_first', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu', kernel_initializer='normal'))
    model.add(Dense(units=50, activation='relu', kernel_initializer='normal'))
    model.add(Dense(units=num_classes, activation='softmax', kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model 

# bulid model 
model = CNN_model() 

# enabling tensorboard 
tensorboard = TensorBoard(log_dir="./graph_CNN", histogram_freq=0, write_graph=True, write_images=True)
tensorboard.set_model(model) 

# fit the model
model.fit(X_train, y_train, batch_size=16, epochs=10, callbacks=[tensorboard],
          validation_data=(X_test, y_test), verbose=1)

# evaluating model 
scores = model.evaluate(X_test, y_test, verbose=1)
print("\nCNN model error %.2f%%" % (100 - scores[1] * 100))