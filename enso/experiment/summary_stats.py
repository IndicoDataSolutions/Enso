"""Module for any LR-style experiment."""
import pandas as pd
import tensorflow as tf
import numpy as np

from enso.experiment.attention import BaseAttnClassifier



class SummaryStatsClassifier(BaseAttnClassifier):

    def reset(self):
        tf.reset_default_graph()
        self.input = None
        self.logits = None
        self.unique_labels = None
        self.probas = None
        self.embed_dim = None
        self.default_keep_prob = 0.7
        self.g = tf.Graph()

    def _build_model(self, embed_dim, num_classes):
        with self.g.as_default():
            self.input = tf.placeholder(tf.float32, shape=[None, None, embed_dim])
            self.targets = tf.placeholder(tf.int32, shape=[None])
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            max_embeddings = tf.reduce_max(self.input, axis=1)
            mean_embeddings = tf.reduce_mean(self.input, axis=1)
            _mean = tf.expand_dims(mean_embeddings, axis=1)
            devs_squared = tf.square(self.input - _mean)
            var_embeddings = tf.reduce_mean(devs_squared, axis=1)

            summary_weights = tf.get_variable('summary_weights', shape=(3,), dtype=tf.float32, initializer=tf.ones_initializer())
            summary_biases = tf.get_variable('summary_biases', shape=(3,), dtype=tf.float32, initializer=tf.zeros_initializer())

            # learned sensitivity to different summary statistics
            embeddings = (
                summary_weights[0] * (max_embeddings + summary_biases[0]) +
                summary_weights[1] * (mean_embeddings + summary_biases[1]) +
                summary_weights[2] * (var_embeddings + summary_biases[2])
            )
            embeddings = tf.nn.dropout(embeddings, keep_prob=self.keep_prob)
            self.logits = tf.contrib.layers.fully_connected(embeddings, num_classes, activation_fn=None)
            self.probas = tf.nn.softmax(self.logits)
