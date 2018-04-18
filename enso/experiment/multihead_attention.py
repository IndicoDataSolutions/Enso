"""Module for any LR-style experiment."""
import pandas as pd
import tensorflow as tf
import numpy as np

from enso.experiment.attention import BaseAttnClassifier

REG_STRENGTH = 50.


class MultiheadAttnClassifier(BaseAttnClassifier):

    param_grid = {'l2': [0.0001, 0.001, 0.01, 0.1], 'n_heads': [1, 2, 4]}

    def __init__(self, l2, l2_attn, n_heads, **kwargs):
        self.l2 = l2
        self.n_heads = n_heads

    def reset(self):
        tf.reset_default_graph()
        self.input = None
        self.logits = None
        self.unique_labels = None
        self.probas = None
        self.embed_dim = None
        self.default_keep_prob = 0.7
        self.g = tf.Graph()

    def _build_model(self, embed_dim, n_classes, n_examples):
        """
        Number of parameters:
        n_hidden * n_heads * n_classes
        """
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2)

        with self.g.as_default():
            # Implementation shouldn't use conv2d, doable with just matrix mults
            self.input = tf.placeholder(tf.float32, shape=[None, None, embed_dim]) # (batch, seq_len, embed_dim)
            self.targets = tf.placeholder(tf.int32, shape=[None]) # (batch,)
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.lengths = tf.placeholder(tf.int32, shape=[None]) # (batch,)
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            flattened_inp = tf.reshape(self.input, shape=[-1, embed_dim]) # (batch * seq_len, embed_dim)
            attention_keys = tf.contrib.layers.fully_connected(
                flattened_inp,
                self.n_heads,
                activation_fn=None,
                weights_regularizer=regularizer
            ) # (batch * seq_len, n_heads)

            attention_logits = tf.reshape(attention_keys, tf.concat([tf.shape(self.input)[:-1], [self.n_heads]], axis=0)) # (batch, seq_len, n_heads)
            attention_weights = tf.nn.softmax(attention_logits, dim=1) # (batch, seq_len, n_heads)
            transposed_weights = tf.transpose(attention_weights, [0, 2, 1]) # (batch, n_heads, seq_len)
            attended_embeddings = tf.matmul(transposed_weights, self.input) # (batch, n_heads, embed_dim)
            head_logits = tf.get_variable('head_logits', shape=(1, self.n_heads, 1), dtype=tf.float32) # (1, n_heads, 1)
            head_weights = tf.nn.softmax(head_logits)
            attended_embeddings = tf.reduce_sum(attended_embeddings * head_weights, axis=1) # (batch, embed_dim)

            self.logits = tf.contrib.layers.fully_connected(
                attended_embeddings,
                n_classes,
                activation_fn=None,
                weights_regularizer=regularizer
            )
            self.probas = tf.nn.softmax(self.logits)


class MultiheadAttnV2Classifier(BaseAttnClassifier):

    def reset(self):
        tf.reset_default_graph()
        self.input = None
        self.logits = None
        self.unique_labels = None
        self.probas = None
        self.embed_dim = None
        self.default_keep_prob = 0.7
        self.g = tf.Graph()

    def _build_model(self, embed_dim, n_classes, n_heads=10):
        """
        Number of parameters:
        n_embed * n_heads
        """
        with self.g.as_default():
            # Implementation shouldn't use conv2d, doable with just matrix mults
            self.input = tf.placeholder(tf.float32, shape=[None, None, embed_dim]) # (batch, seq_len, embed_dim)
            self.targets = tf.placeholder(tf.int32, shape=[None]) # (batch,)
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            regularizer = tf.contrib.layers.l2_regularizer(scale=REG_STRENGTH)

            flattened_inp = tf.reshape(self.input, shape=[-1, embed_dim]) # (batch * seq_len, embed_dim)
            attn_kernel = tf.get_variable('dot_attention', shape=(embed_dim, n_heads), dtype=tf.float32)

            regularizer(attn_kernel)

            attention_keys = tf.matmul(flattened_inp, attn_kernel) # (batch * seq_len, n_heads)
            attention_logits = tf.reshape(attention_keys, tf.concat([tf.shape(self.input)[:-1], [n_heads]], axis=0)) # (batch, seq_len, n_heads)
            attention_weights = tf.nn.softmax(attention_logits, dim=1) # (batch, seq_len, n_heads)
            transposed_weights = tf.transpose(attention_weights, [0, 2, 1]) # (batch, n_heads, seq_len)
            attended_embeddings = tf.matmul(transposed_weights, self.input) # (batch, n_heads, embed_dim)

            head_weights = tf.get_variable('attn_head_weights', shape=(1, n_heads, 1))
            softmaxed_head_weights = tf.nn.softmax(head_weights, dim=1)
            weighted_embeddings = tf.reduce_sum(tf.multiply(softmaxed_head_weights, attended_embeddings), axis=1) # (batch, embed_dim)

            self.logits = tf.contrib.layers.fully_connected(weighted_embeddings, n_classes, activation_fn=None, weights_regularizer=regularizer)
            self.probas = tf.nn.softmax(self.logits)
