#!/usr/bin/env python3
"""
9-transformer_encoder.py
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder for transformer architecture.
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Constructor.

        Args:
            N: Number of encoder blocks
            dm: Dimensionality of the model
            h: Number of attention heads
            hidden: Number of hidden units in FFN
            input_vocab: Size of input vocabulary
            max_seq_len: Maximum sequence length
            drop_rate: Dropout rate
        """
        super(Encoder, self).__init__()

        # Public instance attributes
        self.N = N
        self.dm = dm

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)

        # Positional encoding (pre-calculated numpy array)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List of encoder blocks
        self.blocks = []
        for _ in range(N):
            self.blocks.append(
                EncoderBlock(dm, h, hidden, drop_rate)
            )

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, input_seq_len) containing input tokens
            training: Boolean for training mode
            mask: Mask for attention

        Returns:
            Tensor of shape (batch, input_seq_len, dm) containing encoder
            output
        """
        seq_len = tf.shape(x)[1]

        # 1. Apply embedding layer to input tokens
        x = self.embedding(x)  # (batch, input_seq_len, dm)

        # 2. Scale embeddings by sqrt(dm) - common practice
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # 3. Add positional encoding (only for the actual sequence length)
        x += self.positional_encoding[:seq_len, :]

        # 4. Apply dropout
        x = self.dropout(x, training=training)

        # 5. Pass through each encoder block
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x  # (batch, input_seq_len, dm)
