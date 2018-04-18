"""Module for any LR-style experiment."""
import pandas as pd
import tensorflow as tf
import numpy as np

from enso.experiment.attention import BaseAttnClassifier


class MulticlassDotAttnClassifier(BaseAttnClassifier):

    def reset(self):
        tf.reset_default_graph()
        self.input = None
        self.logits = None
        self.unique_labels = None
        self.probas = None
        self.hidden_size = 1
        self.default_keep_prob = 0.7
        self.g = tf.Graph()

    def _build_model(self, embed_dim, n_classes):
        with self.g.as_default():
            # Implementation shouldn't use conv2d, doable with just matrix mults
            self.input = tf.placeholder(tf.float32, shape=[None, None, embed_dim]) # (batch, seq_len, embed_dim)
            self.targets = tf.placeholder(tf.int32, shape=[None]) # (batch,)
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            flatted_inp = tf.reshape(self.input, shape=[-1, embed_dim]) # (batch * seq_len, embed_dim)
            attn_kernel = tf.get_variable('dot_attention', shape=(embed_dim, n_classes), dtype=tf.float32)
            attention_keys = tf.matmul(flatted_inp, attn_kernel) # (batch * seq_len, n_classes)
            attention_logits = tf.reshape(attention_keys, tf.concat([tf.shape(self.input)[:-1], [n_classes]], axis=0)) # (batch, seq_len, n_classes)
            attention_weights = tf.transpose(tf.nn.softmax(attention_logits, dim=1), [0, 2, 1]) # (batch, n_classes, seq_len)
            attended_embeddings = tf.matmul(attention_weights, self.input) # (batch, n_classes, embed_dim)
            attended_embeddings = tf.nn.dropout(attended_embeddings, keep_prob=self.keep_prob)

            # Tie weights
            cls_templates = tf.transpose(attn_kernel, [1, 0])

            # # Leave weights untied
            # cls_templates = tf.get_variable('cls_template', shape=(n_classes, embed_dim), dtype=tf.float32)

            self.logits = tf.reduce_sum(tf.multiply(attended_embeddings, cls_templates), axis=-1)
            self.probas = tf.nn.softmax(self.logits)
