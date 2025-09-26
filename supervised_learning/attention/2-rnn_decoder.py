#!/usr/bin/env python3
"""
Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer
to decode for machine translation:

Class constructor def __init__(self, vocab, embedding, units, batch):
vocab is an integer representing the size of the output vocabulary
embedding is an integer representing the dimensionality of the embedding vector
units is an integer representing the number of hidden units in the RNN cell
batch is an integer representing the batch size
Sets the following public instance attributes:
embedding - a keras Embedding layer that converts words from the vocabulary
into an embedding vector
gru - a keras GRU layer with units units
Should return both the full sequence of outputs as well as the last hidden
state
Recurrent weights should be initialized with glorot_uniform
F - a Dense layer with vocab units
Public instance method def call(self, x, s_prev, hidden_states):
x is a tensor of shape (batch, 1) containing the previous word in the target
sequence as an index of the target vocabulary
s_prev is a tensor of shape (batch, units) containing the previous decoder
hidden state
hidden_states is a tensor of shape (batch, input_seq_len, units)containing
the outputs of the encoder
You should use SelfAttention = __import__('1-self_attention').SelfAttention
You should concatenate the context vector with x in that order
Returns: y, s
y is a tensor of shape (batch, vocab) containing the output word as a one hot
vector in the target vocabulary
s is a tensor of shape (batch, units) containing the new decoder hidden state
$ cat 2-main.py
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        """Constructor"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=False,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)
        self.batch = batch

    def call(self, x, s_prev, hidden_states):
        """Forward pass"""
        # 1. Calcular el vector de contexto usando atención
        context, _ = self.attention(s_prev, hidden_states)

        # 2. Pasar x por la capa de embedding
        x_embedded = self.embedding(x)  # Shape: (batch, 1, embedding)
        x_embedded = tf.reshape(
            x_embedded, (x_embedded.shape[0], x_embedded.shape[2])
            )

        # 3. Concatenar contexto con x_embedded (en ese orden)
        # context shape: (batch, units), x_embedded shape: (batch, embedding)
        gru_input = tf.concat([context, x_embedded], axis=-1)
        gru_input = tf.expand_dims(gru_input, 1)

        # 4. Pasar por la GRU
        output, s = self.gru(gru_input, initial_state=s_prev)
        # output shape: (batch, units), s shape: (batch, units)

        # 5. Pasar por la capa densa para obtener las probabilidades
        y = self.F(output)  # (batch, vocab)

        return y, s
