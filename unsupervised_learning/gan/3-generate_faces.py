#!/usr/bin/env python3
"""3-generate_faces.py"""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds convolutional generator
    and discriminator models for face generation
    """

    def get_generator():
        """Generator model that takes latent vector and generates faces"""
        # Input layer
        inputs = keras.Input(shape=(16,))

        # Dense layer
        x = keras.layers.Dense(2048, activation='tanh')(inputs)

        # Reshape to 2x2x512
        x = keras.layers.Reshape((2, 2, 512))(x)

        # First upsampling
        x = keras.layers.UpSampling2D()(x)

        # First convolution
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Second upsampling
        x = keras.layers.UpSampling2D()(x)  # 8x8x64

        # Second convolutio
        x = keras.layers.Conv2D(16, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Third upsampling
        x = keras.layers.UpSampling2D()(x)  # 16x16x16

        # Third convolution
        x = keras.layers.Conv2D(1, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """Discriminator model that classifies real vs fake faces"""
        # Input layer
        inputs = keras.Input(shape=(16, 16, 1))

        # First convolution
        x = keras.layers.Conv2D(32, kernel_size=3, padding='same')(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Second convolution
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)  # 4x4x64
        x = keras.layers.Activation('tanh')(x)

        # Third convolution
        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)  # 2x2x128
        x = keras.layers.Activation('tanh')(x)

        # Fourth convolution
        x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = keras.layers.MaxPooling2D()(x)  # 1x1x256
        x = keras.layers.Activation('tanh')(x)

        # Flatten and final dense layer
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
