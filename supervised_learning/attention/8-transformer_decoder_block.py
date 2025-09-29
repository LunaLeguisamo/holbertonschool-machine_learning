#!/usr/bin/env python3
"""
8-transformer_decoder_block.py
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Decoder block for transformer architecture.
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
        super(DecoderBlock, self).__init__()

        # Primer Multi-Head Attention (autoatención enmascarada)
        self.mha1 = MultiHeadAttention(dm, h)

        # Segundo Multi-Head Attention (cross-attention)
        self.mha2 = MultiHeadAttention(dm, h)

        # Feed Forward Network
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, target_seq_len, dm) - decoder input
            encoder_output: Tensor of shape (batch, input_seq_len, dm)
            - encoder output
            training: Boolean for training mode
            look_ahead_mask: Mask for first attention (autoregressive)
            padding_mask: Mask for second attention (padding)

        Returns:
            Tensor of shape (batch, target_seq_len, dm)
        """
        # 1. Masked Multi-Head Attention (Self-Attention en decoder)
        # Self-attention enmascarada
        attn1_output, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1_output = self.dropout1(attn1_output, training=training)

        # 2. First Add & Norm
        out1 = self.layernorm1(x + attn1_output)  # Residual connection

        # 3. Multi-Head Attention (Cross-Attention)
        # Q: decoder output, K-V: encoder output
        attn2_output, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
            )
        attn2_output = self.dropout2(attn2_output, training=training)

        # 4. Second Add & Norm
        out2 = self.layernorm2(out1 + attn2_output)  # Residual connection

        # 5. Feed Forward Network
        ffn_output = self.dense_hidden(out2)  # (batch, target_seq_len, hidden)
        # (batch, target_seq_len, dm)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)

        # 6. Third Add & Norm
        output = self.layernorm3(out2 + ffn_output)  # Residual connection

        return output
