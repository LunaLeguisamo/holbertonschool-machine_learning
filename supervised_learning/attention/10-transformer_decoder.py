#!/usr/bin/env python3
"""
10-transformer_decoder.py
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Decoder for transformer architecture.
    """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Constructor.

        Args:
            N: Number of decoder blocks
            dm: Dimensionality of the model
            h: Number of attention heads
            hidden: Number of hidden units in FFN
            target_vocab: Size of target vocabulary
            max_seq_len: Maximum sequence length
            drop_rate: Dropout rate
        """
        super(Decoder, self).__init__()

        # Public instance attributes
        self.N = N
        self.dm = dm

        # Embedding layer for target vocabulary
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)

        # Positional encoding (pre-calculated numpy array)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List of decoder blocks
        self.blocks = []
        for _ in range(N):
            self.blocks.append(
                DecoderBlock(dm, h, hidden, drop_rate)
            )

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, target_seq_len) containing target tokens
            encoder_output: Tensor of shape (batch, input_seq_len, dm)
            from encoder
            training: Boolean for training mode
            look_ahead_mask: Mask for first attention (autoregressive)
            padding_mask: Mask for second attention (padding)

        Returns:
            Tensor of shape (batch, target_seq_len, dm)
            containing decoder output
        """
        seq_len = tf.shape(x)[1]

        # 1. Apply embedding layer to target tokens
        x = self.embedding(x)  # (batch, target_seq_len, dm)

        # 2. Scale embeddings by sqrt(dm) - common practice
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 3. Add positional encoding (only for the actual sequence length)
        x += self.positional_encoding[:seq_len, :]

        # 4. Apply dropout
        x = self.dropout(x, training=training)

        # 5. Pass through each decoder block
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)

        return x  # (batch, target_seq_len, dm)
