#!/usr/bin/env python3
"""
aaa
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Construye un bloque residual con proyección (shortcut convolucional).

    Parámetros:
    - A_prev: Tensor de entrada (salida de la capa anterior)
    - filters: tupla de 3 enteros (F11, F3, F12) que indican:
        * F11: filtros de la primera conv 1x1 (reducción de canales)
        * F3:  filtros de la conv central 3x3
        * F12: filtros de la última conv 1x1 (restauración de canales)
        y también de la shortcut
    - s: stride que se aplica en la primera convolución y en la shortcut

    Proceso:
    - Ruta principal: conv1x1 → BN → ReLU → conv3x3 → BN → ReLU → conv1x1 → BN
    - Shortcut: conv1x1 con stride s → BN
    - Suma ambas rutas → ReLU

    Retorna:
    - Activación final (ReLU) luego de la suma.
    """

    # Asignamos los filtros
    F11, F3, F12 = filters

    # Inicializador HeNormal con semilla 0 (ideal para ReLU)
    he_init = K.initializers.HeNormal(seed=0)

    # ============================
    # 🛣️ Ruta Principal (Main Path)
    # ============================

    # Primera convolución 1x1 con stride s (puede reducir resolución)
    X = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s, padding='same',
                        kernel_initializer=he_init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)  # Normalizamos por canales
    X = K.layers.Activation('relu')(X)

    # Segunda convolución 3x3 (stride=1, mantiene tamaño)
    X = K.layers.Conv2D(filters=F3, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Tercera convolución 1x1 (para expandir a F12 canales)
    X = K.layers.Conv2D(filters=F12, kernel_size=1, strides=1, padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # ===============================
    # 🔁 Ruta Shortcut (con proyección)
    # ===============================

    shortcut = K.layers.Conv2D(
        filters=F12, kernel_size=1, strides=s, padding='same',
        kernel_initializer=he_init
        )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # ================================
    # ➕ Suma de las rutas y activación
    # ================================

    X = K.layers.Add()([X, shortcut])  # Sumamos las dos rutas
    X = K.layers.Activation('relu')(X)  # Activación final

    return X
