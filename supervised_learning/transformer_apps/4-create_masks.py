#!/usr/bin/env python3
"""
4-create_masks.py
"""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation - all in one function
    """
    batch_size = tf.shape(inputs)[0]
    seq_len_in = tf.shape(inputs)[1]
    seq_len_out = tf.shape(target)[1]

    # 1. Encoder padding mask
    encoder_padding_mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
    encoder_mask = encoder_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # 2. Decoder padding mask (for encoder outputs)
    decoder_mask = encoder_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # 3. Combined mask for decoder 1st attention block
    # Lookahead mask
    lookahead_mask = 1 - tf.linalg.band_part(tf.ones((
        seq_len_out, seq_len_out
        )), -1, 0)
    lookahead_mask = lookahead_mask[tf.newaxis, tf.newaxis, :, :]

    # Target padding mask
    target_padding_mask = tf.cast(tf.not_equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Combine: if either mask has 1 (masked), then mask that position
    combined_mask = tf.maximum(lookahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
