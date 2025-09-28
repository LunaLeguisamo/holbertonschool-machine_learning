#!/usr/bin/env python3
"""
7-transformer_encoder_block.py
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block for transformer architecture.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Constructor.

        Args:
            dm: Dimensionality of the model
            h: Number of attention heads
            hidden: Number of hidden units in FFN
            drop_rate: Dropout rate
        """
        super(EncoderBlock, self).__init__()

        # Multi-Head Attention
        self.mha = MultiHeadAttention(dm, h)

        # Feed Forward Network (2 capas)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, input_seq_len, dm)
            training: Boolean for training mode
            mask: Mask for attention

        Returns:
            Tensor of shape (batch, input_seq_len, dm)
        """
        # 1. Multi-Head Attention sub-layer
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)

        # 2. First Add & Norm (residual connection + layer norm)
        out1 = self.layernorm1(x + attn_output)  # Residual connection

        # 3. Feed Forward Network
        ffn_output = self.dense_hidden(out1)  # (batch, seq_len, hidden)
        ffn_output = self.dense_output(ffn_output)  # (batch, seq_len, dm)
        ffn_output = self.dropout2(ffn_output, training=training)

        # 4. Second Add & Norm
        output = self.layernorm2(out1 + ffn_output)  # Residual connection

        return output
