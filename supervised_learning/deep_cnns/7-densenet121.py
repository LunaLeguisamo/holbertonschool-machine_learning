#!/usr/bin/env python3
"""
Build the DenseNet-121 architecture
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds DenseNet-121 with fixed layer names."""
    he_init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Initial Convolution (Name layers explicitly)
    X = K.layers.BatchNormalization()(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=2 * growth_rate,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=he_init,
        use_bias=False)(X)
    X = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding='same')(X)

    # Dense Block 1 (6 layers)
    X, nb_filters = dense_block(X, 2 * growth_rate, growth_rate, 6)

    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2 (12 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3 (24 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4 (16 layers)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final layers
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)  # Fixed naming
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=he_init)(X)

    model = K.models.Model(inputs=X_input, outputs=X)
    return model
