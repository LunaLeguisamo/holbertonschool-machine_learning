#!/usr/bin/env python3
"""
4-create_masks.py
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation of transformer model

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) - input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out) - target sentence

    Returns:
        encoder_mask: padding mask for encoder (batch_size, 1, 1, seq_len_in)
        combined_mask: combined mask for decoder 1st attention block
                     (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: padding mask for decoder 2nd attention block
                     (batch_size, 1, 1, seq_len_in)
    """
    # 1. Encoder padding mask
    encoder_padding_mask = tf.cast(tf.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # 2. Decoder padding mask
    decoder_mask = encoder_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # 3. Combined mask for decoder 1st attention block
    seq_len_out = tf.shape(target)[1]
    lookahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
        )
    lookahead_mask = lookahead_mask[tf.newaxis, tf.newaxis, :, :]

    target_padding_mask = tf.cast(tf.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Combine masks: take maximum
    combined_mask = tf.maximum(lookahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
