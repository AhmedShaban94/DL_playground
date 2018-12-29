import keras 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.layers import Dense, Dropout 
from keras.models import Model, Sequential 
from keras.datasets import mnist 
from keras.optimizers import Adam 
from keras.layers.advanced_activations import LeakyReLU
from tqdm import tqdm  

# Add xrange iterator 
import sys

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

# make random seed fixed number 
np.random.seed(10)

# Dimension of random_noise vector 
random_noise_dim = 100 

def load_mnist_data():
    #load data 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #normalize data to be between [-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5 )/ 127.5
    #reshape data to be (60000, 784)
    x_train = x_train.reshape(60000, 784) 
    return (x_train, y_train, x_test, y_test) 

def get_generator(): 
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=random_noise_dim))
    generator.add(LeakyReLU())
    
    generator.add(Dense(512))
    generator.add(LeakyReLU())
    
    generator.add(Dense(1024))
    generator.add(LeakyReLU())

    generator.add(Dense(784, activation='tanh')) 
    generator.compile(loss=keras.losses.binary_crossentropy
                      , optimizer=keras.optimizers.Adam())
    return generator 

def get_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(1024))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512)) 
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))
        
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(1, activation=keras.activations.sigmoid))
    discriminator.compile(loss=keras.losses.binary_crossentropy
                          , optimizer=Adam())
    return discriminator 


def gan_network(random_noise_dim, discriminator, generator):
    # We just train discriminator or generator one at atime 
    discriminator.trainable = False 
    
    gan_input = keras.layers.Input(shape=(random_noise_dim,))
    
    # output of generator 
    x = generator(gan_input)
    
    # get output of discriminator and network  
    gan_output = discriminator(x)
    
    gan_model = Model(inputs=gan_input, outputs=gan_output)
    gan_model.compile(loss=keras.losses.binary_crossentropy, 
                      optimizer=Adam())
    
    return gan_model


# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_noise_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)
    
def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train, y_train, x_test, y_test = load_mnist_data()
    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] / batch_size

    # Build our GAN netowrk
    generator = get_generator()
    discriminator = get_discriminator()
    gan = gan_network(random_noise_dim, discriminator, generator)

    for e in xrange(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(xrange(int(batch_count))):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_noise_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_noise_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)

if __name__ == '__main__':
    train(400, 128)

    