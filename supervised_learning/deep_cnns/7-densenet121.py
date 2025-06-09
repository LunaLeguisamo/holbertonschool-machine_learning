#!/usr/bin/env python3
"""dfgfdf"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    dfsdfdsf
    """
    input = K.Input(shape=(224, 224, 3))
    he_init = K.initializers.he_normal(seed=0)

    X = K.layers.BatchNormalization(axis=3)(input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=he_init)(X)
    X = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(X)

    X, nb_filters = dense_block(X, 64, growth_rate, 6)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.AveragePooling2D(pool_size=(7, 7))(X)

    op = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=he_init)(X)

    model = K.Model(inputs=input, outputs=op)
    return model
