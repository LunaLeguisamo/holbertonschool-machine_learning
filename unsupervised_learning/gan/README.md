In this task we will define a class WGAN_clip and test it on some examples. Almost all of what we need is already explained in the presentation or in task 0. We just have to modify the losses of the generator and the discriminator, and to clip the weights of the discriminator in [-1,1].

The code to be filled in

Basically we keep the same structure :

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class WGAN_clip(keras.Model) :
    
    def __init__( self, generator , discriminator , latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        super().__init__()                         # run the __init__ of keras.Model first. 
        self.latent_generator = latent_generator
        self.real_examples    = real_examples
        self.generator        = generator
        self.discriminator    = discriminator
        self.batch_size       = batch_size
        self.disc_iter        = disc_iter
        
        self.learning_rate    = learning_rate
        self.beta_1=.5                               # standard value, but can be changed if necessary
        self.beta_2=.9                               # standard value, but can be changed if necessary
        
        # define the generator loss and optimizer:
        self.generator.loss      = lambda x : pass           # <----- new !
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer , loss=generator.loss )
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss      = lambda x,y : pass   # <----- new !
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer , loss=discriminator.loss )
       
    
    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size :
            size= self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size :
            size= self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices  = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
        
             
    # overloading train_step()    
    def train_step(self,useless_argument):
        pass
        #for _ in range(self.disc_iter) :
            
            # compute the loss for the discriminator in a tape watching the discriminator's weights
                # get a real sample
                # get a fake sample
                # compute the loss discr_loss of the discriminator on real and fake samples
            # apply gradient descent once to the discriminator
            
            # clip the weights (of the discriminator) between -1 and 1    # <----- new !

        # compute the loss for the generator in a tape watching the generator's weights 
            # get a fake sample 
            # compute the loss gen_loss of the generator on this sample
        # apply gradient descent to the generator
        
        
        
        # return {"discr_loss": discr_loss, "gen_loss": gen_loss}
Your task:

It consists in filling in three holes :

fill in the generator_loss function in the __init__ method
fill in the discriminator_loss function in the __init__ method
fill in the train_step method
Reminder:

In the presentation we have seen that for a Wasserstein GAN,

generator_loss(x) is the opposite of the mean value of the image x of the discriminator on the image by the generator of a batch of latent vectors
discriminator_loss(x,y) is the difference between
the mean value of the image y of a batch of real examples by the discriminator and
the mean value of the image x by the discriminator of the image by the generator of a batch of latent vectors
The weights of the discriminator must be clipped in [-1,1].
Hints:

to compute the losses you should use the function tf.math.reduce_mean
note that list discriminator.trainable_variables is a list containing the weights and the biases under the form of tensors
to clip a tensor you should use the function tf.clip_by_value
Main

$ cat 0-main.py

#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt

WGAN_clip = __import__('1-wgan_clip').WGAN_clip


## Regulating the seed

def set_seeds(seed) :
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def hash_tensor(tensor) :
    return np.sum(np.array([hash(x)%2**30 for x in tensor.numpy().flatten()]))%2**30


def spheric_generator(nb_points, dim) :
    u=tf.random.normal(shape=(nb_points, dim))
    return u/tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(u),axis=[1])+10**-8),[nb_points,1])

def normal_generator(nb_points, dim) :
    return tf.random.normal(shape=(nb_points, dim))

def uniform_generator(nb_points, dim) :
    return tf.random.uniform(shape=(nb_points, dim))


## Building Generator and Discriminator

def fully_connected_GenDiscr(gen_shape, real_examples, latent_type="normal" ) :

    #   Latent generator
    if latent_type   == "uniform" :
        latent_generator  =  lambda k : uniform_generator(k,gen_shape[0])
    elif latent_type == "normal" :
        latent_generator  =  lambda k : normal_generator(k,gen_shape[0])
    elif latent_type == "spheric" :
        latent_generator  =  lambda k : spheric_generator(k,gen_shape[0])

    #   Generator
    inputs     = keras.Input(shape=( gen_shape[0] , ))
    hidden     = keras.layers.Dense( gen_shape[1] , activation = 'tanh'    )(inputs)
    for i in range(2,len(gen_shape)-1) :
        hidden = keras.layers.Dense( gen_shape[i] , activation = 'tanh'    )(hidden)
    outputs    = keras.layers.Dense( gen_shape[-1], activation = 'sigmoid' )(hidden)
    generator  = keras.Model(inputs, outputs, name="generator")

    #   Discriminator
    inputs        = keras.Input(shape=( gen_shape[-1], ))
    hidden        = keras.layers.Dense( gen_shape[-2],   activation = 'tanh' )(inputs)
    for i in range(2,len(gen_shape)-1) :
        hidden    = keras.layers.Dense( gen_shape[-1*i], activation = 'tanh' )(hidden)
    outputs       = keras.layers.Dense( 1 ,              activation = 'tanh' )(hidden)
    discriminator = keras.Model(inputs, outputs, name="discriminator")

    return generator, discriminator, latent_generator

# illustration:
def illustr_fully_connected_GenDiscr() :
    generator, discriminator, latent_generator = fully_connected_GenDiscr([1,100,100,2], None)
    print(generator.summary())
    print(discriminator.summary())


## Training a fully connected GAN (Simple GAN)

def example_fully_connected_GAN(chosen_type ,real_examples, gen_shape, epochs,  batch_size=200, steps_per_epoch=250, latent_type="normal", learning_rate=.005):
    generator, discriminator, latent_generator = fully_connected_GenDiscr(gen_shape, real_examples, latent_type=latent_type )
    G=chosen_type( generator , discriminator , latent_generator, real_examples, learning_rate=learning_rate)
    G.compile()
    G.fit(real_examples, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    return G


## An example of a large real sample

def four_clouds_example(N_real):
    X   = np.random.randn( N_real)*.05
    Y   = np.random.randn( N_real)*.05
    X[:N_real//2]+=.75
    X[N_real//2:]+=.25
    Y[N_real//4:N_real//2]+=.25
    Y[:N_real//4]+=.75
    Y[N_real//2:3*N_real//4]+=.75
    Y[3*N_real//4:]+=.25
    R   = np.minimum(X*X,1)
    G   = np.minimum(Y*Y,1)
    B   = np.maximum(1-R-G,0)
    return tf.convert_to_tensor(np.vstack([X,Y,R,G,B]).T)


## Visualize 5D the result of G

def visualize_5D(G, show=True, title=None, filename=None, dpi=200):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flatten():
        ax.set_xlim(-.1, 1.1)
        ax.set_ylim(-.1, 1.1)

    X_tf = G.real_examples
    X = X_tf.numpy()
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=X[:, 2:], s=1)
    axes[0, 0].set_title("real")

    lat = G.latent_generator(10000)
    Y_tf = G.generator(lat)
    Y = Y_tf.numpy()
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], c=Y[:, 2:], s=1)
    axes[0, 1].set_title("fake = generator(latent sample) ")

    cX = G.discriminator(X_tf).numpy()
    cY = G.discriminator(Y_tf).numpy()
    m = min(np.min(cX), np.min(cY))
    M = max(np.max(cX), np.max(cY))

    axes[1, 0].scatter(X[:, 0], X[:, 1], c=cX, s=1, vmin=m, vmax=M)
    axes[1, 0].set_title("discriminator(real)")

    Y = G.generator(G.latent_generator(10000)).numpy()
    axes[1, 1].scatter(Y[:, 0], Y[:, 1], c=cY, s=1, vmin=m, vmax=M)
    axes[1, 1].set_title("discriminator(fake)")

    for i in range(2):
        axes[i, 2].set_xlim(-3, 3)
        axes[i, 2].set_ylim(-3, 3)

    A = np.linspace(-3, 3, 150)
    B = np.linspace(-3, 3, 150)
    X, Y = np.meshgrid(A, B)
    U = tf.convert_to_tensor(np.vstack([X.ravel(), Y.ravel()]).T)

    X = G.discriminator(G.generator(U)).numpy()
    u = axes[0, 2].pcolormesh(A, B, X[:, 0].reshape([150, 150]), shading='gouraud')

    axes[0, 2].set_title(r"discriminator $\circ$ generator on latent space")

    axes[1, 2].scatter(lat.numpy()[:, 0], lat.numpy()[:, 1], c=cY, s=1)
    axes[1, 2].set_title(r"discriminator $\circ$ generator on latent sample")
    # sci = lambda x : "{:.2E}".format(.99*x*(x>0)+1.01*x*(x<0))
    # cb=fig.colorbar(u, ax=axes[1])
    ##cb.set_ticks([sci(np.min(X[:,0])), 0, sci(np.max(X[:,0]))])
    # cb.ax.tick_params(labelsize=5)

    # some more tuning:
    for a in axes.flatten():
        a.tick_params(axis='both', which='major', labelsize=5)
        a.tick_params(axis='both', which='minor', labelsize=5)

    if title:
        fig.suptitle(title)

    if filename:
        plt.savefig(filename, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    elif show:
        plt.show()

## LET'S GO !

set_seeds(0)
G = example_fully_connected_GAN(WGAN_clip ,four_clouds_example(1000), [2,10,10,5], 16, steps_per_epoch=100, learning_rate=.001)
visualize_5D(G,show=True, title=f"after 16 epochs")


$ ./0-main.py
should produce the following picture:

