#!/usr/bin/env python3
"""
LeNet-5 modified architecture using Keras
"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras
    Args:
        X: K.Input of shape (m, 28, 28, 1)
    Returns:
        A K.Model compiled to use Adam optimization and accuracy metrics
    """
    he = K.initializers.HeNormal(seed=0)

    # First convolutional layer (6 filters, 5x5, same padding)
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=he)(X)

    # First max pooling layer (2x2, stride 2)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)

    # Second convolutional layer (16 filters, 5x5, valid padding)
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer=he)(pool1)

    # Second max pooling layer (2x2, stride 2)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)

    # Flatten the output
    flat = K.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    fc1 = K.layers.Dense(units=120,
                         activation='relu',
                         kernel_initializer=he)(flat)

    # Fully connected layer with 84 nodes
    fc2 = K.layers.Dense(units=84,
                         activation='relu',
                         kernel_initializer=he)(fc1)

    # Output layer with 10 nodes (softmax)
    output = K.layers.Dense(units=10,
                            activation='softmax',
                            kernel_initializer=he)(fc2)

    # Define and compile the model
    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
