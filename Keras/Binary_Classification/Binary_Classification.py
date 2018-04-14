import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7 
np.random.seed(seed)

# reading dataset 
data = pd.read_csv('sonar.csv', header=None).values  

# data preprocessing 
X = data[:,0:60].astype(float)
Y = data[:,60] 
encoder = LabelEncoder()
Y_labeld = encoder.fit_transform(Y)

# building model
def base_function(): 
    model = Sequential()
    model.add(Dense(units=120, input_dim=60, kernel_initializer='uniform', activation='relu')) 
    model.add(Dense(units=240, kernel_initializer='uniform', activation='relu')) 
    model.add(Dense(units=120, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model 

# training 
estimator = KerasClassifier(build_fn=base_function, epochs=100, batch_size=5, verbose=0) 
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('mlp', KerasClassifier(build_fn=base_function, epochs=100, batch_size=5 ,verbose=0))) 
pipeline = Pipeline(estimators) 
fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed) 
results = cross_val_score(pipeline, X, Y_labeld, cv=fold) 
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# smaller model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y_labeld, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(30, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y_labeld, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))





