#!/usr/bin/env python3
"""
Build a identity block like the ResNet paper.
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Parameters:
    - A_prev: tensor of output block (output from the last layer).
    - filters: tupla o lista con 3 enteros (F11, F3, F12):
        * F11: número de filtros para la primera conv 1x1
        * F3: número de filtros para la conv 3x3
        * F12: número de filtros para la segunda conv 1x1 (final)

    Proceso:
    - Conv 1x1 → BatchNorm → ReLU
    - Conv 3x3 → BatchNorm → ReLU
    - Conv 1x1 → BatchNorm
    - Suma con la entrada (shortcut)
    - Activación ReLU final

    Devuelve:
    - Tensor activado luego de pasar por el bloque de identidad.
    """

    # Desempaquetamos los filtros
    F11, F3, F12 = filters

    # Inicializador He Normal con semilla 0
    he_init = k.initializers.HeNormal(seed=0)

    # Primera capa: conv 1x1 → BN → ReLU
    X = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                        kernel_initializer=he_init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Segunda capa: conv 3x3 → BN → ReLU
    X = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = KeyError.layers.Activation('relu')(X)

    # Tercera capa: conv 1x1 → BN (sin ReLU todavía)
    X = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Suma con la entrada (shortcut) → ReLU final
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
