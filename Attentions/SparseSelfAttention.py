# --------------- Import Necessary Packages -----------
import tensorflow as tf
from keras.layers import *



class SparseSelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SparseSelfAttention, self).__init__(**kwargs)
        self.value_dense = None
        self.key_dense = None
        self.query_dense = None
        self.hidden_dim = None

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]
        self.query_dense = Dense(self.hidden_dim)
        self.key_dense = Dense(self.hidden_dim)
        self.value_dense = Dense(self.hidden_dim)
        super(SparseSelfAttention, self).build(input_shape)

    def call(self, inputs):

        # Query, Key, value layers
        Q = self.query_dense(inputs)
        K = self.key_dense(inputs)
        V = self.value_dense(inputs)

        # Compute attention scores
        attention_scores = tf.matmul(Q, K, transpose_b=True)

        # for getting all attention scores
        k = tf.shape(attention_scores)[-1]
        top_k_scores, _ = tf.nn.top_k(attention_scores, k=k)
        threshold = tf.expand_dims(top_k_scores[:, :, -1], axis=-1)
        sparse_attention_scores = tf.where(attention_scores >= threshold, attention_scores, tf.fill(tf.shape(attention_scores), -1e9))

        # Apply softmax to the sparse scores
        sparse_attention_weights = tf.nn.softmax(sparse_attention_scores, axis=-1)

        # Weighted sum of the values
        attended_output = tf.matmul(sparse_attention_weights, V)  # [batch_size, seq_len, hidden_dim]

        return attended_output

    def compute_output_shape(self, input_shape):
        return input_shape, (input_shape[0], input_shape[1], input_shape[1])


"""
Attention_layer = SparseSelfAttention()(layer)
"""
