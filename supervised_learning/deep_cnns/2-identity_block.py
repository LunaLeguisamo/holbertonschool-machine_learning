#!/usr/bin/env python3
"""
a
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    a
    """

    # Desempaquetamos los filtros
    F11, F3, F12 = filters

    # Inicializador He Normal con semilla 0
    he_init = K.initializers.HeNormal(seed=0)

    # Primera capa: conv 1x1 → BN → ReLU
    X = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                        kernel_initializer=he_init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Segunda capa: conv 3x3 → BN → ReLU
    X = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Tercera capa: conv 1x1 → BN (sin ReLU todavía)
    X = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Suma con la entrada (shortcut) → ReLU final
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
