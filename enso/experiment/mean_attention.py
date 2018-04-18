"""Module for any LR-style experiment."""
import pandas as pd
import tensorflow as tf
import numpy as np

from enso.experiment.attention import BaseAttnClassifier


class ReduceMeanClassifier(BaseAttnClassifier):

    param_grid = {'l2': [1e-4, 1e-3, 1e-2, 1e-1]}

    def __init__(self, l2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2 = l2

    def reset(self):
        tf.reset_default_graph()
        self.input = None
        self.logits = None
        self.unique_labels = None
        self.probas = None
        self.embed_dim = None
        self.default_keep_prob = 0.7
        self.g = tf.Graph()

    def _build_model(self, embed_dim, num_classes, n_examples):
        with self.g.as_default():
            self.input = tf.placeholder(tf.float32, shape=[None, None, embed_dim]) # (batch, seq, embed_dim)
            self.targets = tf.placeholder(tf.int32, shape=[None]) # (batch,)
            self.lengths = tf.placeholder(tf.int32, shape=[None]) # (batch,)
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2)

            scaled_embeddings = self.input / tf.to_float(tf.reshape(self.lengths, [-1, 1, 1]))
            mean_embeddings = tf.reduce_sum(scaled_embeddings, axis=1)
            # sum_embeddings = tf.reduce_sum(self.input, axis=1) # (batch, embed_dim)

            mean_embeddings = tf.reduce_mean(self.input, axis=1) # (batch, embed_dim)

            self.logits = tf.contrib.layers.fully_connected(
                mean_embeddings, num_classes, activation_fn=None, weights_regularizer=regularizer
            )
            self.probas = tf.nn.softmax(self.logits)
