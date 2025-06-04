#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate


def inception_block(A_prev, filters):
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv_1x1 = Conv2D(filters=F1, kernel_size=(1, 1),
                      padding='same', activation='relu')(A_prev)

    # 1x1 followed by 3x3 convolution branch
    conv_3x3_reduce = Conv2D(filters=F3R, kernel_size=(1, 1),
                             padding='same', activation='relu')(A_prev)
    conv_3x3 = Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                      activation='relu')(conv_3x3_reduce)

    # 1x1 followed by 5x5 convolution branch
    conv_5x5_reduce = Conv2D(filters=F5R, kernel_size=(1, 1),
                             padding='same', activation='relu')(A_prev)
    conv_5x5 = Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                      activation='relu')(conv_5x5_reduce)

    # Max pooling followed by 1x1 convolution branch
    max_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                            padding='same')(A_prev)
    conv_pool_proj = Conv2D(filters=FPP, kernel_size=(1, 1),
                            padding='same', activation='relu')(max_pool)

    # Concatenate all branches
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, conv_pool_proj],
                         axis=-1)

    return output
