"""Module for any LR-style experiment."""
import pandas as pd
import tensorflow as tf
import numpy as np

from enso.experiment.attention import BaseAttnClassifier


class ReduceMaxClassifier(BaseAttnClassifier):

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

            # weight elmo embeddings
            if embed_dim == 3072:
                layer1, layer2, layer3 = tf.split(self.input, 3, axis=2)
                layer_weights = tf.get_variable('summary_weights', shape=(3,), dtype=tf.float32, initializer=tf.ones_initializer())
                softmax_layer_weights = tf.nn.softmax(layer_weights)
                _input = (
                    softmax_layer_weights[0] * layer1 +
                    softmax_layer_weights[1] * layer2 +
                    softmax_layer_weights[2] * layer3
                )
                flattened_inp = tf.reshape(_input, shape=[-1, embed_dim // 3])
            else:
                _input = self.input
                flattened_inp = tf.reshape(_input, shape=[-1, embed_dim])

            max_embeddings = tf.reduce_max(self.input, axis=1)
            max_embeddings = tf.nn.dropout(max_embeddings, keep_prob=self.keep_prob)
            self.logits = tf.contrib.layers.fully_connected(max_embeddings, num_classes, activation_fn=None)
            self.probas = tf.nn.softmax(self.logits)
