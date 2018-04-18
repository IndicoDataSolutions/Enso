"""Module for any LR-style experiment."""
import pandas as pd
import tensorflow as tf
import numpy as np

from enso.experiment.attention import BaseAttnClassifier


class FancyL2DotAttnClassifier(BaseAttnClassifier):

    param_grid = {'l2': [0.0001, 0.001, 0.01], 'l2_attn': [0.1, 1.0]}

    def __init__(self, l2, l2_attn, **kwargs):
        super().__init__(**kwargs)
        self.l2 = l2
        self.l2_attn = l2_attn

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
        with self.g.as_default():
            # Implementation shouldn't use conv2d, doable with just matrix mults
            self.input = tf.placeholder(tf.float32, shape=[None, None, embed_dim], name='inputs') # (batch, seq_len, embed_dim)
            self.lengths = tf.placeholder(tf.int32, shape=[None], name='lengths') # batch
            self.targets = tf.placeholder(tf.int32, shape=[None], name='targets') # (batch,)
            self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.additional_loss_terms = []

            regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2)
            attn_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_attn)

            flattened_inp = tf.reshape(self.input, shape=[-1, embed_dim]) # (batch * seq_len, embed_dim)
            attention_keys = tf.contrib.layers.fully_connected(
                flattened_inp,
                1,
                activation_fn=None,
                weights_regularizer=attn_regularizer
            )
            attention_logits = tf.reshape(attention_keys, tf.shape(self.input)[:-1]) # (batch, seq_len)

            attention_weights = tf.nn.softmax(attention_logits, dim=-1) # (batch, seq_len)
            mask = tf.sequence_mask(self.lengths, dtype=tf.float32)

            # ensure model does not attend to zero vectors
            masked_attention_weights = attention_weights * mask

            reshaped_weights = tf.expand_dims(masked_attention_weights, 1) # (batch, 1, seq_len)

            attended_embeddings = tf.matmul(reshaped_weights, self.input) # (batch, 1, embed_dim)
            flattened_embeddings = tf.squeeze(attended_embeddings, axis=1) # (batch, embed_dim)
            self.logits = tf.contrib.layers.fully_connected(
                flattened_embeddings,
                n_classes,
                activation_fn=None,
                weights_regularizer=regularizer
            )
            self.probas = tf.nn.softmax(self.logits)
