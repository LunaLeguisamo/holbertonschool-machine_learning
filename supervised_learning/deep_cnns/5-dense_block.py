#!/usr/bin/env python3
"""
Build a dense block
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional Networks
    
    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block
        
    Returns:
        The concatenated output of each layer within the Dense Block
        The number of filters within the concatenated outputs
    """
    for i in range(layers):
        # Batch Normalization followed by ReLU activation
        X1 = K.layers.BatchNormalization()(X)
        X1 = K.layers.Activation('relu')(X1)
        
        # Bottleneck layer: 1x1 conv to reduce feature maps (DenseNet-B)
        X1 = K.layers.Conv2D(filters=4 * growth_rate,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=K.initializers.he_normal(seed=0))(X1)
        
        # Batch Normalization followed by ReLU activation
        X1 = K.layers.BatchNormalization()(X1)
        X1 = K.layers.Activation('relu')(X1)
        
        # 3x3 convolution
        X1 = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=K.initializers.he_normal(seed=0))(X1)
        
        # Concatenate the input with the new feature maps
        X = K.layers.concatenate([X, X1])
        
        # Update the number of filters
        nb_filters += growth_rate
    
    return X, nb_filters
