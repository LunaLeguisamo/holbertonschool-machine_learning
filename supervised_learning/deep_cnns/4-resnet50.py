#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture using identity and projection blocks.
"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 model
    without pretrained weights"""
    model = K.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=1000
    )
    return model
