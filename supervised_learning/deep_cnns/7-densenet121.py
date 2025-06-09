#!/usr/bin/env python3
"""
Build the DenseNet-121 architecture
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture
    as described in the paper:
    https://arxiv.org/pdf/1608.06993.pdf
    """
    he_init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Initial convolution and pooling (Conv1 + Pool1)
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=he_init,
        name='conv1/conv')(X_input)

    X = K.layers.BatchNormalization(axis=3, name='conv1/bn')(X)
    X = K.layers.Activation('relu', name='conv1/relu')(X)
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool1')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 64, growth_rate, 6)

    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final BatchNorm + ReLU
    X = K.layers.BatchNormalization(axis=3, name='bn')(X)
    X = K.layers.Activation('relu', name='relu')(X)

    # Global Average Pooling
    X = K.layers.GlobalAveragePooling2D(name='avg_pool')(X)

    # Dense (softmax) classifier
    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=he_init,
        name='fc1000')(X)

    # Model
    model = K.Model(inputs=X_input, outputs=output, name="DenseNet121")

    return model
