from keras.layers import Dense 
from keras.models import Sequential 
from keras.datasets import imdb 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

seed = 7 
np.random.seed(seed)
 
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = np.concatenate((X_train, y_train)) 
Y = np.concatenate((X_test, y_test)) 
