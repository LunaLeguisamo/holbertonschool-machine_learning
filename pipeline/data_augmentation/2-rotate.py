#!/usr/bin/env python3
"""
Write a function  that rotates an image by 90 degrees counter-clockwise:

image is a 3D tf.Tensor containing the image to rotate
Returns the rotated image
"""
import tensorflow as tf


def rotate_image(image):
    """
    image is a 3D tf.Tensor containing the image to rotate
    Returns the rotated image
    """
    rotated_image = tf.image.rot90(image)
    return rotated_image
