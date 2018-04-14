import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder 
from time import time 

time1 = time() 
seed = 7 
np.random.seed(seed) 

dataframe = pd.read_csv('iris.csv', header=None)
dataset = dataframe.values 
X = dataset[:, :-1].astype(np.float32)
Y = dataset[:, -1] 

encoder = LabelEncoder() 
encoded_Y = encoder.fit_transform(Y)  
dummy_Y = np_utils.to_categorical(encoded_Y) 

def base_model():
    model = Sequential()
    model.add(Dense(units=8, activation='relu', input_dim=4))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model 

estimator = KerasClassifier(build_fn=base_model, epochs=200, batch_size=5, verbose=1) 
fold = KFold(shuffle=True, n_splits=10, random_state=seed)

results = cross_val_score(estimator, X, dummy_Y, cv=fold) 
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))  
print("Time taken = ", time() - time1) 
    
