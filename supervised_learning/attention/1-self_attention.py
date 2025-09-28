import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    """SelfAttention class"""

    def __init__(self, units):
        """Class constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units, kernel_initializer='glorot_uniform')
        self.U = tf.keras.layers.Dense(units, kernel_initializer='glorot_uniform')
        self.V = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')

    def call(self, s_prev, hidden_states):
        """Calculates the attention for machine translation"""
        # s_prev shape: (batch, units)
        # hidden_states shape: (batch, input_seq_len, units)
        
        # Expand s_prev to (batch, 1, units) for broadcasting
        s_prev_expanded = tf.expand_dims(s_prev, 1)
        
        # Calculate W(s_prev) and U(hidden_states)
        W_s_prev = self.W(s_prev_expanded)  # (batch, 1, units)
        U_hidden = self.U(hidden_states)    # (batch, input_seq_len, units)
        
        # Sum with broadcasting
        combined = tf.nn.tanh(W_s_prev + U_hidden)
        
        # Calculate score
        score = self.V(combined)  # (batch, input_seq_len, 1)
        
        # Calculate attention weights
        weights = tf.nn.softmax(score, axis=1)
        
        # Calculate context vector
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        
        return context, weights
    