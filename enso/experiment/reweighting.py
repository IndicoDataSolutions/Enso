import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from enso.experiment import ClassificationExperiment
from enso.config import RESULTS_DIRECTORY, GOLD_FRAC
from enso.registry import Registry, ModeKeys
import numpy as np


class TFEstimatorExperiment(ClassificationExperiment):
    estimator = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_enc = LabelEncoder()
        self.model = None

    def fit(self, X, y):
        y = self.label_enc.fit_transform(y)
        num_gold = int(GOLD_FRAC * len(y))
        x_train_gold = X[:num_gold]
        y_train_gold = y[:num_gold]
        self.model = tf.estimator.Estimator(model_fn=self.estimator,
                                            params={
                                                "n_class": len(self.label_enc.classes_),
                                                "gold": (np.array(x_train_gold, dtype=np.float32), np.array(y_train_gold, dtype=np.int32))
                                            })
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X), y=np.asarray(y), num_epochs=None, shuffle=True)
        self.model.train(input_fn=train_input_fn, steps=5000)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X, dtype=np.float32), num_epochs=1, shuffle=False)
        preds = self.model.predict(input_fn=predict_input_fn)
        return pd.DataFrame.from_records(
            {c: prob for c, prob in zip(self.label_enc.classes_, p["prob"])} for p in preds)


def model(net, n_classes, dropout=True):
    for i, units in enumerate([n_classes * 5, n_classes * 2]):
        w = tf.get_variable("W_{}".format(i), shape=[net.get_shape().as_list()[-1], units], dtype=tf.float32)
        b = tf.get_variable("b_{}".format(i), shape=[units], dtype=tf.float32)
        net = tf.nn.xw_plus_b(net, w, b)
        net = tf.nn.relu(net)
        if dropout:
            net = tf.layers.dropout(net, 0.1)
    # Compute logits (1 per class).
    return tf.layers.dense(net, n_classes, activation=None)


def smax_xent_loss(output, targets):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(targets, depth=tf.shape(output)[-1]), logits=output)


def my_model_learn_to_reweight(self, features, labels, mode, params):
    net = features
    goldx, goldy = params["gold"]
    goldx = tf.constant(goldx, dtype=tf.float32)
    goldy = tf.constant(goldy, dtype=tf.int32)

    def local_model(input):
        return model(input, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits, loss = learning_to_reweight(goldx, goldy, net, labels, local_model, smax_xent_loss, lr=0.1)
    else:
        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            logits = local_model(net)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Compute evaluation metrics.
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    loss = tf.reduce_mean(smax_xent_loss(output=logits, targets=labels))
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


def my_model_baseline(self, features, labels, mode, params):
    net = features

    logits = model(net, n_classes=params["n_class"],  dropout=mode == tf.estimator.ModeKeys.TRAIN)
    # Compute predictions.

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.reduce_mean(smax_xent_loss(output=logits, targets=labels))

    # Create training op.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Compute evaluation metrics.
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


@Registry.register_experiment(ModeKeys.CLASSIFY,
                              requirements=[
                                  ("Featurizer", "not PlainTextFeaturizer"),
                                  ("Resampler", "CorruptiveResampler")
                              ])
class Reweighting(TFEstimatorExperiment):
    estimator = my_model_learn_to_reweight


@Registry.register_experiment(ModeKeys.CLASSIFY,
                              requirements=[
                                  ("Featurizer", "not PlainTextFeaturizer"),
                                  ("Resampler", "CorruptiveResampler")
                              ])
class ReweightingBaseline(TFEstimatorExperiment):
    estimator = my_model_baseline


def learning_to_reweight(gold_data, gold_targets, data, targets, model, loss, lr=1e-5):
    # lines 405 initial forward pass to compute the initial weighted loss
    def net(data):
        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            return model(data)

    def meta_net(data):
        with tf.variable_scope("meta_net", reuse=tf.AUTO_REUSE):
            return model(data)

    # Lines 4 - 5 initial forward pass to compute the initial weighted loss
    y_f_hat = net(data)
    y_f_hat_meta = meta_net(data)

    net_vars = tf.global_variables(scope="net")
    meta_net_vars = tf.global_variables(scope="meta_net")
    re_init_vars = []
    for n_v, met_v in zip(net_vars, meta_net_vars):
        re_init_vars.append(met_v.assign(n_v))

    with tf.control_dependencies(re_init_vars):
        cost = loss(y_f_hat_meta, targets)
    eps = tf.zeros_like(cost)
    l_f_meta = tf.reduce_sum(cost * eps)

    # Line 6 perform a parameter update
    grads = tf.gradients(l_f_meta, meta_net_vars)
    patch_dict = dict()
    for grad, var in zip(grads, meta_net_vars):
        if grad is None:
            print("None grad for variable {}".format(var.name))
        else:
            patch_dict[var.name] = -grad * lr

    # Monkey patch get_variable
    old_get_variable = tf.get_variable

    def _get_variable(*args, **kwargs):
        var = old_get_variable(*args, **kwargs)
        return var + patch_dict.get(var.name, 0.0)

    tf.get_variable = _get_variable

    # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
    y_g_hat = meta_net(gold_data)

    tf.get_variable = old_get_variable

    l_g_meta = loss(y_g_hat, gold_targets)

    grad_eps_es = tf.gradients(l_g_meta, eps)[0]

    # Line 11 computing and normalizing the weights
    w_tilde = tf.maximum(-grad_eps_es, 0.)
    norm_c = tf.reduce_sum(w_tilde)

    w = w_tilde / (norm_c + tf.cast(tf.equal(norm_c, 0.0), dtype=tf.float32))

    # Lines 12 - 14 computing for the loss with the computed weights
    # and then perform a parameter update
    cost = loss(y_f_hat, targets)
    l_f = tf.reduce_sum(cost * w)

    logits = y_f_hat
    loss = l_f

    return logits, loss
