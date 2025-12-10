#!/usr/bin/env python3
"""
Write a function that performs a random crop of an image:
image is a 3D tf.Tensor containing the image to crop
size is a tuple containing the size of the crop
Returns the cropped image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    image is a 3D tf.Tensor containing the image to crop
    size is a tuple containing the size of the crop
    Returns the cropped image
    """
    cropped_image = tf.image.random_crop(image, size)
    return cropped_image
