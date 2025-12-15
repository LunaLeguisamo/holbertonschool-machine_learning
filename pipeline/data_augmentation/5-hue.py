#!/usr/bin/env python3
"""
Write a function that changes the hue of an image
image is a 3D tf.Tensor containing the image to change
delta is the amount the image hue should be changed
Returns the altered image
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an RGB image.

    Args:
        image: 3D tf.Tensor of shape (H, W, 3), dtype float32 or uint8.
               If uint8, it will be converted to float32 in [0,1].
        delta: Amount to change the hue, float in [-1, 1].

    Returns:
        The hue-adjusted image as a tf.Tensor, same shape as input.
    """
    # tf.image.adjust_hue expects floats in [0, 1]
    if image.dtype == tf.uint8:
        image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply hue adjustment
    image = tf.image.adjust_hue(image, delta)

    return image
