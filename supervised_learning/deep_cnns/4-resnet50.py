#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture using identity and projection blocks.
"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture.

    Returns:
        keras.Model: ResNet-50 model
    """
    init = K.initializers.HeNormal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Initial convolution and max pooling
    X = K.layers.Conv2D(64, kernel_size=7, strides=2, padding='same',
                        kernel_initializer=init)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Conv2_x
    X = projection_block(X, [64, 64, 256], s=1)
    for _ in range(2):
        X = identity_block(X, [64, 64, 256])

    # Conv3_x
    X = projection_block(X, [128, 128, 512], s=2)
    for _ in range(3):
        X = identity_block(X, [128, 128, 512])

    # Conv4_x
    X = projection_block(X, [256, 256, 1024], s=2)
    for _ in range(5):
        X = identity_block(X, [256, 256, 1024])

    # Conv5_x
    X = projection_block(X, [512, 512, 2048], s=2)
    for _ in range(2):
        X = identity_block(X, [512, 512, 2048])

    # Average Pooling and output layer
    X = K.layers.AveragePooling2D(pool_size=7, strides=1)(X)
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(X)

    return K.Model(inputs=X_input, outputs=X)
