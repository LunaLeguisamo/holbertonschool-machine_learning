#!/usr/bin/env python3
"""
6-multihead_attention.py
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention layer for transformers.
    """
    def __init__(self, dm, h):
        """
        Constructor.

        Args:
            dm: Dimensionality of the model
            h: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h  # número de heads
        self.dm = dm  # dimensionalidad del modelo

        # Verificar que dm es divisible por h
        assert dm % h == 0, "dm must be divisible by h"

        # Profundidad de cada head
        self.depth = dm // h

        # Capas Dense para proyecciones
        self.Wq = tf.keras.layers.Dense(dm)  # para queries
        self.Wk = tf.keras.layers.Dense(dm)  # para keys
        self.Wv = tf.keras.layers.Dense(dm)  # para values
        self.linear = tf.keras.layers.Dense(dm)  # capa lineal final

    def split_heads(self, x, batch_size):
        """
        Divide el último eje en (h, depth) y transpone para atención.

        Args:
            x: Tensor de forma (batch_size, seq_len, dm)
            batch_size: Tamaño del batch

        Returns:
            Tensor de forma (batch_size, h, seq_len, depth)
        """
        # Reshape a (batch_size, seq_len, h, depth)
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))

        # Transponer a (batch_size, h, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Forward pass.

        Args:
            Q: Tensor (batch, seq_len_q, dk)
            K: Tensor (batch, seq_len_v, dk)
            V: Tensor (batch, seq_len_v, dv)
            mask: Tensor o None

        Returns:
            output: Tensor (batch, seq_len_q, dm)
            weights: Tensor (batch, h, seq_len_q, seq_len_v)
        """
        batch_size = tf.shape(Q)[0]

        # 1. Aplicar proyecciones lineales
        Q = self.Wq(Q)  # (batch, seq_len_q, dm)
        K = self.Wk(K)  # (batch, seq_len_v, dm)
        V = self.Wv(V)  # (batch, seq_len_v, dm)

        # 2. Dividir en múltiples heads
        Q = self.split_heads(Q, batch_size)  # (batch, h, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch, h, seq_len_v, depth)
        V = self.split_heads(V, batch_size)  # (batch, h, seq_len_v, depth)

        # 3. Aplicar atención escalada por producto punto a cada head
        # scaled_attention shape: (batch, h, seq_len_q, depth)
        # attention_weights shape: (batch, h, seq_len_q, seq_len_v)
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)

        # 4. Concatenar los heads
        # Transponer de (batch, h, seq_len_q, depth)
        # a (batch, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenar: (batch, seq_len_q, h * depth) = (batch, seq_len_q, dm)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # 5. Aplicar capa lineal final
        output = self.linear(concat_attention)  # (batch, seq_len_q, dm)

        return output, attention_weights
