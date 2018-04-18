"""Module for any LR-style experiment."""
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib.opt import ScipyOptimizerInterface
from enso.experiment import Experiment


class BaseAttnClassifier(Experiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_loss_terms = []

    def _pad_data(self, training_data):
        max_seq_len = max([len(seq) for seq in training_data])
        for i, seq in enumerate(training_data):

            pad_diff = max_seq_len - len(seq)
            if pad_diff:
                diff_arr = np.zeros((pad_diff, self.embed_dim))
                training_data[i] = np.vstack((seq, diff_arr))
            else:
                training_data[i] = np.asarray(seq)

        input_embeddings = np.asarray(training_data)
        return input_embeddings

    def train(self, training_data, training_labels):
        self.reset()
        with self.g.as_default():
            for i, data in enumerate(training_data):
                if not isinstance(data, np.ndarray):
                    training_data[i] = data['vectors']

                # stripping 0 vectors
                arr = [vec for vec in training_data[i] if np.sum(vec) != 0.]
                if arr:
                    arr = np.asarray(arr)
                else:
                    arr = np.zeros((1, 300))
                training_data[i] = np.asarray(arr)

            n_examples = len(training_labels)
            self.unique_labels = list(set(training_labels))
            num_classes = len(self.unique_labels)

            self.embed_dim = len(training_data[0][0])
            input_labels = np.asarray([self.unique_labels.index(label) for label in training_labels])

            if self.input is None:

                self._build_model(self.embed_dim, num_classes, n_examples=n_examples)

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.reg_loss = tf.reduce_sum(reg_losses)
                self.mean_loss = tf.reduce_mean(loss)
                self.total_loss = self.mean_loss + self.reg_loss

                # LBFGS
                self.training_op = ScipyOptimizerInterface(self.total_loss, method='L-BFGS-B')

                config = tf.ConfigProto(device_count={'GPU': 0})
                self.session = tf.Session(graph=self.g, config=config)
                self.session.run(tf.global_variables_initializer())

            n_data_points = len(training_data)
            for i in range(1):
                batch_start, batch_size = 0, len(input_labels)
                # for batch_start in range(0, len(input_labels), batch_size):
                batch_end = batch_start + batch_size
                training_data_batch = training_data[batch_start:batch_end]
                lengths = [len(ex) for ex in training_data_batch]
                input_embeddings_batch = self._pad_data(training_data_batch)
                input_labels_batch = input_labels[batch_start:batch_end]
                self.training_op.minimize(self.session, feed_dict={
                    self.targets: input_labels_batch,
                    self.input: input_embeddings_batch,
                    self.keep_prob: self.default_keep_prob,
                    self.is_training: True,
                    self.lengths: lengths
                })

    def predict(self, test_data, subset='TEST'):
        with self.g.as_default():
            for i, data in enumerate(test_data):
                if isinstance(data, dict):
                    test_data[i] = data['vectors']

                # stripping 0 vectors
                arr = [vec for vec in test_data[i] if np.sum(vec) != 0.]
                if arr:
                    arr = np.asarray(arr)
                else:
                    arr = np.zeros((1, 300))
                test_data[i] = np.asarray(arr)

            batch_size = 128
            class_probas = []
            for batch_start in range(0, len(test_data), batch_size):
                batch_end = batch_start + batch_size
                test_data_batch = test_data[batch_start:batch_end]
                input_embeddings_batch = self._pad_data(test_data_batch)
                lengths = [len(ex) for ex in test_data_batch]
                class_probas_batch = self.session.run(self.probas, feed_dict={
                    self.input: input_embeddings_batch,
                    self.keep_prob: 1.0,
                    self.is_training: False,
                    self.lengths: lengths
                })
                class_probas.extend(class_probas_batch.tolist())

            class_probas = np.asarray(class_probas)
            df = pd.DataFrame({label: class_probas[:, i] for i, label in enumerate(self.unique_labels)})

            return df


class RegularizedMLPAttnClassifier(BaseAttnClassifier):

    param_grid = {'l2': [0.001, 0.01, 0.1], 'n_hidden': [10]}

    def __init__(self, l2, n_hidden, *args, **kwargs):
        self.l2 = l2
        self.n_hidden = n_hidden

    def reset(self):
        tf.reset_default_graph()
        self.input = None
        self.logits = None
        self.unique_labels = None
        self.probas = None
        self.hidden_size = self.n_hidden
        self.embed_dim = None
        self.default_keep_prob = 0.7
        self.g = tf.Graph()

    def _build_model(self, embed_dim, num_classes, non_lin=tf.tanh, n_examples=None):
        with self.g.as_default():
            self.input = tf.placeholder(tf.float32, shape=[None, None, embed_dim])
            self.targets = tf.placeholder(tf.int32, shape=[None])
            self.lengths = tf.placeholder(tf.int32, shape=[None])
            self.keep_prob = tf.placeholder(tf.float32, shape=[])
            self.is_training = tf.placeholder(tf.bool, name='is_training')

            regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2)

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

            # flattened_inp = tf.nn.dropout(flattened_inp, keep_prob=self.keep_prob)
            hidden_state = tf.contrib.layers.fully_connected(
                flattened_inp,
                self.hidden_size,
                activation_fn=non_lin,
                weights_regularizer=regularizer
            )
            attention_keys = tf.contrib.layers.fully_connected(
                hidden_state, 1, activation_fn=None
            )
            reshaped_keys = tf.reshape(attention_keys, tf.shape(self.input)[:-1])
            reshaped_keys = tf.expand_dims(reshaped_keys, 1)
            keys = tf.nn.softmax(reshaped_keys, dim=-1)
            attended_embeddings = tf.matmul(keys, self.input)
            flattened_embeddings = tf.squeeze(attended_embeddings, axis=1)
            self.logits = tf.contrib.layers.fully_connected(
                flattened_embeddings,
                num_classes,
                activation_fn=None,
                weights_regularizer=regularizer
            )
            self.probas = tf.nn.softmax(self.logits)
