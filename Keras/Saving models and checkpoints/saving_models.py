from keras.models import Sequential, model_from_json 
from keras.layers import Dense 
import numpy as np 
import pandas as pd 

seed = 7 
np.random.seed(seed)

# reading data 
data = pd.read_csv('pima-indians-diabetes.csv', header=None).values 
X = data[:, 0:9]
Y = data[:, 9] 

# building model 
model = Sequential()
model.add(Dense(units=9, input_dims=9, activation='relu', kernel_initializer='uniform')) 
model.add(Dense(units=4, activation='relu', kernel_initializer='normal'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal'))

model.compile(optimizer='adam', loss='binary-crossentropy', metrics=['accuarcy']) 
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
scores = model.evaluate(X, Y, 10, 150, 0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 

# you can save the whole model (architecture + weights + optimizer state)
mode.save('model.h5') 

# you can load the model also 
from keras.models import load_model 
model = load_model('model.h5')

# to save only the architecture of the model use JSON or YAML file 
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)  