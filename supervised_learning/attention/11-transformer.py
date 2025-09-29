#!/usr/bin/env python3
"""
11-transformer.py
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Constructor.

        Args:
            N: Number of blocks in encoder and decoder
            dm: Dimensionality of the model
            h: Number of attention heads
            hidden: Number of hidden units in FFN
            input_vocab: Size of input vocabulary
            target_vocab: Size of target vocabulary
            max_seq_input: Maximum input sequence length
            max_seq_target: Maximum target sequence length
            drop_rate: Dropout rate
        """
        super(Transformer, self).__init__()

        # Public instance attributes
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, input_seq_len) - source sequence
            target: Tensor of shape (batch, target_seq_len) - target sequence
            training: Boolean for training mode
            encoder_mask: Padding mask for encoder
            look_ahead_mask: Look-ahead mask for decoder self-attention
            decoder_mask: Padding mask for decoder cross-attention

        Returns:
            Tensor of shape (batch, target_seq_len, target_vocab)
            - output logits
        """
        # 1. Encode the input sequence
        enc_output = self.encoder(inputs, training, encoder_mask)

        # 2. Decode the target sequence using encoder output
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # 3. Project to target vocabulary size
        final_output = self.linear(dec_output)

        return final_output
