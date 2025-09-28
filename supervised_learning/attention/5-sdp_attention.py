#!/usr/bin/env python3
"""
5-sdp_attention.py
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q: tensor with shape (..., seq_len_q, dk)
        K: tensor with shape (..., seq_len_v, dk)
        V: tensor with shape (..., seq_len_v, dv)
        mask: tensor broadcastable to (..., seq_len_q, seq_len_v)

    Returns:
        output: tensor with shape (..., seq_len_q, dv)
        weights: tensor with shape (..., seq_len_q, seq_len_v)
    """
    # 1. Calcular Q·Kᵀ
    # (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # 2. Obtener dk de la dimensión de Q
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # 3. Escalar los scores
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 4. Aplicar máscara si existe
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 5. Calcular pesos de atención (softmax en el último eje)
    # (..., seq_len_q, seq_len_v)
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 6. Aplicar a los values
    output = tf.matmul(weights, V)  # (..., seq_len_q, dv)

    return output, weights
