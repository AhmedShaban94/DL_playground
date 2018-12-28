import keras 
import numpy as np 
import gzip 
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Activation
from keras.optimizers import RMSprop 
from sklearn.model_selection import train_test_split 

# extract data from gzip file 
def extract_data(file_name, num_images):
    with gzip.open(file_name) as bytestream: 
        bytestream.read(16) 
        buffer = bytestream.read(28 * 28 * num_images) 
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) 
        data = data.reshape(num_images, 28, 28)
        return data 

# extract labels from gzip file 
def extract_labels(file_name, num_images):
    with gzip.open(file_name) as bytestream: 
        bytestream.read(8)
        buffer = bytestream.read(1 * num_images) 
        lables = np.frombuffer(buffer, dtype=np.uint8).astype(np.float64) 
        return lables 


NUM_IMAGES_TRAIN = 60000 
NUM_IMAGES_TEST = 10000 
    
# extract data 
train_data = extract_data('dataset/train/train-images-idx3-ubyte.gz', NUM_IMAGES_TRAIN)
test_data = extract_data('dataset/test/t10k-images-idx3-ubyte.gz', NUM_IMAGES_TEST)
   
# extract labels 
train_labels = extract_labels('dataset/train/train-labels-idx1-ubyte.gz', NUM_IMAGES_TRAIN)
test_labels = extract_labels('dataset/test/t10k-labels-idx1-ubyte.gz', NUM_IMAGES_TEST) 
       
# Create Dictionary of target classes 
label_dict = {
 0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
}

# Reshape train data and test data to be in 28 * 28 *1 Matrix  
train_data = train_data.reshape(-1, 28, 28, 1) 
test_data =  test_data.reshape(-1, 28, 28, 1)

# resacling data to max value of pixels in each dataset (0.0 -> 1.0)
train_data = train_data / np.max(train_data) 
test_data = test_data  / np.max(test_data) 

#split training data into train/validation sets 
train_X, valid_X, train_GT, valid_GT = train_test_split(
        train_data,
        train_data, 
        test_size=0.33, 
        random_state=13) 

#training params 
batch_size = 128 
epoch = 50 
num_channel = 1 
x, y = 28, 28 


'''build Model'''  
model = keras.Sequential()

# Build Encoder 

# Conv2D -> 1 
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1))) 
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 

# Conv2D -> 2 
model.add(Conv2D(64, (3, 3), padding='same')) 
model.add(BatchNormalization()) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 

# Conv2D -> 3 
model.add(Conv2D(128, (3, 3), padding='same')) 
model.add(BatchNormalization()) 
model.add(Activation('relu')) 

# Build Decoder 

# Conv2D -> 3
model.add(Conv2D(128, (3, 3), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(UpSampling2D((2, 2))) 

# Conv2D -> 2 
model.add(Conv2D(64, (3, 3), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(UpSampling2D((2,2))) 

# final decoder  
model.add(Conv2D(1, (3, 3), padding='same'))
model.add(Activation('sigmoid'))

# compile model 
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=RMSprop(), 
              metrics=['accuracy']) 

# traing model 
autoencoder_train = model.fit(train_X, train_GT,
                              batch_size=batch_size
                              ,epochs=epoch,verbose=1,
                              validation_data=(valid_X, valid_GT))  

# save model (vanilla AutoEncoder)
model.save('model.h5')

