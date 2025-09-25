#!/usr/bin/env python3
"""
Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer
to calculate the attention for machine translation

Class constructor def __init__(self, units):
units is an integer representing the number of hidden units in the alignment
model
Sets the following public instance attributes:
W - a Dense layer with units units, to be applied to the previous decoder
hidden state
U - a Dense layer with units units, to be applied to the encoder hidden states
V - a Dense layer with 1 units, to be applied to the tanh of the sum of the
outputs of W and U
Public instance method def call(self, s_prev, hidden_states):
s_prev is a tensor of shape (batch, units) containing the previous decoder
hidden state
hidden_states is a tensor of shape (batch, input_seq_len, units)containing
the outputs of the encoder
Returns: context, weights
context is a tensor of shape (batch, units) that contains the context vector
for the decoder
weights is a tensor of shape (batch, input_seq_len, 1) that contains the
attention weights
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """SelfAttention class"""

    def __init__(self, units):
        """Class constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculates the attention for machine translation"""
        # Expand s_prev to (batch, 1, units) to match hidden_states shape
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Calculate score
        score =\
            self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))

        # Calculate attention weights
        weights = tf.nn.softmax(score, axis=1)

        # Calculate context vector as the weighted sum of hidden states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
