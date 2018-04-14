from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint 
import numpy as np 
import pandas as pd 

seed = 7 
np.random.seed(seed)

# reading data 
data = pd.read_csv('pima-indians-diabetes.csv', header=None).values
X = data[:, 0:-1]
Y = data[:, -1] 

model = Sequential()
model.add(Dense(units=8, input_dim=8, activation='relu', kernel_initializer='uniform'))
model.add(Dense(units=9, activation='relu', kernel_initializer='uniform'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform')) 

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# creating checkpoints with best model
filepath = 'weights-best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='max', verbose=1)
callback_list = [checkpoint]
model.fit(X, Y, epochs=120, callbacks=callback_list, validation_split=0.33, verbose=0) 

