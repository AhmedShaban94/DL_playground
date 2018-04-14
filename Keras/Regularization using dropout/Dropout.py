from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD 
from keras.constraints import maxnorm 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import StratifiedKFold, cross_val_score 

seed = 7 
np.random.seed(seed)

data = pd.read_csv("sonar.csv", header=None).values 
X = data[:, :-1].astype(float)
Y = data[:, -1] 

# encoding Y labels 
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y) 

def baseline_model(): 
    # building model 
    model = Sequential()
    model.add(Dense(input_dim=60, units=30, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(units=20, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(units=10, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal'))
    # compiling model 
    sgd = SGD(lr=0.01, momentum=0.8, decay=0, nesterov=False)
    model.compile(optimizer=sgd, metrics=['accuracy'], loss='binary_crossentropy')
    return model 

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, epochs=300, batch_size=16, verbose=1)))

pipeline = Pipeline(estimators)
fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y_encoded, cv=fold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)) 

    
    
    


