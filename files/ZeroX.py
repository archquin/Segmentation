#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:57:31 2022

@author: archquin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 19:21:57 2022
https://keras.io/examples/generative/conditional_gan/
@author: archquin
"""

from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

batch_size = 64
num_channels = 1
num_classes = 25
image_size = 28
latent_dim = 128

dataset=[]
dataseth= []
datasett=[]
datasetht = []

Ldict={1:"α",2:"β",3:"γ",4:"δ",5:"ε",6:"ζ",7:"η",8:"θ",9:"ι",10:"κ",11:"λ",12:"μ",13:"ν",14:"ξ",15:"ο",16:"π",17:"ρ",18:"σ",19:"τ",20:"υ",21:"φ",22:"χ",23:"ψ",24:"ω"}
Rdict = dict((v, k) for k, v in Ldict.items())

def sig(x):
    return 1/(1 + np.exp(-x))



for i in range(1,241):
    data1 = cv2.imread('train_high_resolution/letter_bnw_'+str(i)+'.jpg')
  #  data1 = cv2.imread('train_letters_images/image_'+str(i)+'.jpg')

    img = cv2.resize(data1, (28,28))
    dataset.append(img)


    if i < 11:
        d = 'ω'
    elif i < 21 :
        d = 'ψ'
    elif i < 31 :
        d = 'χ'
    elif i < 41 :
        d = 'φ'
    elif i < 51 :
        d = 'υ'
    elif i < 61 :
        d = 'τ'
    elif i < 71 :
        d = 'σ'
    elif i < 81 :
        d = 'ρ'
    elif i < 91 :
        d = 'π'
    elif i < 101 :
        d = 'ο'
    elif i < 111 :
        d = 'ξ'
    elif i < 121 :
        d = 'ν'
    elif i < 131 :
        d = 'μ'
    elif i < 141 :
        d = 'λ'
    elif i < 151 :
        d = 'κ'
    elif i < 161 :
        d = 'ι'
    elif i < 171 :
        d = 'θ'
    elif i < 181 :
        d = 'η'
    elif i < 191 :
        d = 'ζ'
    elif i < 201 :
        d = 'ε'
    elif i < 211 :
        d = 'δ'
    elif i < 221 :
        d = 'γ'
    elif i < 231 :
        d = 'β'
    elif i <= 241 :
        d = 'α'
        
    dataseth.append(Rdict[d])
    

for i in range(1,97):
    data2 = cv2.imread('test_high_resolution/letter_bnw_test_'+str(i)+'.jpg')
   # data2 = cv2.imread('test_letters_images/image_test_'+str(i)+'.jpg')
    imgt = cv2.resize(data2, (28,28))
    datasett.append(imgt)
    dataset.append(imgt)

    if i <= 4 :
        d = 'α'
    elif i <= 8 :
        d = 'β'
    elif i <= 12 :
        d = 'γ'
    elif i <= 16 :
        d = 'δ'  
    elif i <= 20 :
        d = 'ε'   
    elif i <= 24 :
        d = 'ζ'
    elif i <= 28 :
        d = 'η'
    elif i <= 32 :
        d = 'θ'
    elif i <= 36 :
        d = 'ι'
    elif i <= 40 :
        d = 'κ'
    elif i <= 44 :
        d = 'λ'
    elif i <= 48 :
        d = 'μ'
    elif i <= 52 :
        d = 'ν'
    elif i <= 56 :
        d = 'ξ'
    elif i <= 60 :
        d = 'ο'
    elif i <= 64 :
        d = 'π'
    elif i <= 68 :
        d = 'ρ'
    elif i <= 72 :
        d = 'σ'          
    elif i <= 76 :
        d = 'τ'   
    elif i <= 80 :
        d = 'υ'
    elif i <= 84 :
        d = 'φ'
    elif i <= 88 :
        d = 'χ'
    elif i <= 92 :
        d = 'ψ'
    elif i <= 96:
        d = 'ω'
        
    datasetht.append(Rdict[d])
    dataseth.append(Rdict[d])

    
#https://data-flair.training/blogs/handwritten-character-recognition-neural-network/   
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = np.array(dataset, dtype="float32")
datat =  np.array(datasett, dtype="float32")
data = data[:, :, :, 0]
datat = datat[:, :, :, 0]

'''
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

X = df1.iloc[:, :-1]
X = np.array(X)
X = X.reshape(X.shape[0], 14, 14)
Y = df1.iloc[:, -1:]
Y = np.array(Y)


Xt = df2.iloc[:, :-1]
Xt = np.array(Xt)
Xt = Xt.reshape(Xt.shape[0], 14, 14)

Yt = df2.iloc[:, -1:]
Yt = np.array(Yt)
'''

all_digits = np.concatenate([data])
all_labels = np.concatenate([dataseth])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.

all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 25)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")


generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
      #  layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"),
    #    layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)


class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat(
            [generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat(
            [real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat(
                [fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


Zero = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
Zero.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

Zero.fit(dataset, epochs=1111)
trained_gen = Zero.generator

tf.saved_model.save(Zero,'Zzzero_O')

# Choose the number of intermediate images that would be generated in
# between the interpolation + 2 (start and last images).
num_interpolation = 25  # @param {type:"greek alphabet letter"}

# Sample noise for the interpolation.
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(
    interpolation_noise, (num_interpolation, latent_dim))


def interpolate_class(first_number, second_number):
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # Calculate the interpolation vector between the two labels.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
        first_label * (1 - percent_second_label) +
        second_label * percent_second_label
    )

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat(
        [interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake

def save_plot(examples,n):
        for i in range(n*n):
            plt.subplot(n,n,1+i)
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        plt.show()

for i in range(1,25):
    start_class = i  # @param {type:"slider", min:0, max:9, step:1}
    end_class = i  # @param {type:"slider", min:0, max:9, step:1}

    fake_images = interpolate_class(start_class, end_class)

    fake_images *= 255.0
    img = fake_images.astype(np.uint8)
    img = tf.image.resize(img, (96, 96)).numpy().astype(np.uint8)

    save_plot(img,5)
