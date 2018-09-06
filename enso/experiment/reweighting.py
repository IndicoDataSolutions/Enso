import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from enso.experiment import ClassificationExperiment
from finetune import ReweightingClassifier
from enso.config import RESULTS_DIRECTORY, GOLD_FRAC
from enso.registry import Registry, ModeKeys
from enso.utils import OversampledKFold
import numpy as np
from sklearn.model_selection import GridSearchCV


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class FinetuneReweighting(ClassificationExperiment):
    """
    LanguageModel finetuning as an alternative to simple models trained on top of pretrained features.
    """

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = ReweightingClassifier(val_size=0.0, batch_size=15, max_length=128, low_memory_mode=True)

    def fit(self, X, y):
        """
        :param X: `np.ndarray` of raw text sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        num_gold = int(GOLD_FRAC * len(y))
        x_train_gold = X[:num_gold]
        y_train_gold = y[:num_gold]
        self.model.fit(x_train_gold, y_train_gold, X, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        preds = self.model.predict_proba(X)
        return pd.DataFrame.from_records(preds)


class TFEstimatorExperiment(ClassificationExperiment):
    estimator = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_enc = LabelEncoder()
        self.model = None

    @staticmethod
    def model_body_to_model_fn(model_body):
        def model_fn(features, labels, mode, params):
            goldx, goldy = params["gold"]
            goldx = tf.constant(goldx, dtype=tf.float32)
            goldy = tf.constant(goldy, dtype=tf.int32)

            logits, loss = model_body(goldx, goldy, features, labels, mode, params)

            # Compute predictions.
            predicted_classes = tf.argmax(logits, 1)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class': predicted_classes,
                    'prob': tf.nn.softmax(logits)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            # Create training op.
            loss = tf.reduce_mean(loss)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

        return model_fn

    def fit(self, X, y):
        y = self.label_enc.fit_transform(y)
        num_gold = int(GOLD_FRAC * len(y))
        x_train_gold = X[:num_gold]
        y_train_gold = y[:num_gold]
        n_steps = 5000

        self.model = tf.estimator.Estimator(model_fn=self.model_body_to_model_fn(self.estimator),
                                            params={
                                                "n_class": len(self.label_enc.classes_),
                                                "gold": (np.array(x_train_gold, dtype=np.float32),
                                                np.array(y_train_gold, dtype=np.int32)),
                                                "n_steps": n_steps
                                            })
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X, np.float32), y=np.asarray(y, np.int32),
                                                            num_epochs=None, shuffle=True)
        self.model.train(input_fn=train_input_fn, steps=n_steps)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X, dtype=np.float32), num_epochs=1,
                                                              shuffle=False)
        preds = self.model.predict(input_fn=predict_input_fn)
        return pd.DataFrame.from_records(
            {c: prob for c, prob in zip(self.label_enc.classes_, p["prob"])} for p in preds)


def model(net, n_classes, dropout=True):
    if dropout:
        net = tf.layers.dropout(net, 0.1)
    w = tf.get_variable("W_output", shape=[net.get_shape().as_list()[-1], n_classes], dtype=tf.float32)
    b = tf.get_variable("b_output", shape=[n_classes], dtype=tf.float32)

    return tf.nn.xw_plus_b(net, w, b)


def reweighting_body(_, goldx, goldy, features, labels, mode, params):
    def local_model(input):
        return model(input, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits, loss = learning_to_reweight(goldx, goldy, features, labels, local_model, smax_xent_loss, lr=0.1)
    else:
        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            logits = local_model(features)
            loss = None
    return logits, loss


def baseline_body(_, goldx, goldy, features, labels, mode, params):
    logits = model(features, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = smax_xent_loss(output=logits, targets=labels, n_class=params["n_class"])
    else:
        loss = None

    return logits, loss


def gold_every_batch_body(_, goldx, goldy, features, labels, mode, params):
    logits = model(tf.concat((goldx, features), axis=0), n_classes=params["n_class"],
                   dropout=mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = smax_xent_loss(output=logits, targets=tf.concat((goldy, labels), axis=0), n_class=params["n_class"])
    else:
        loss = None

    return logits[goldy.get_shape()[0]:], loss


def train_on_gold_body(_, goldx, goldy, features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(goldx, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)
        loss = tf.reduce_mean(smax_xent_loss(output=logits, targets=goldy))
    else:
        logits = model(features, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)
        loss = None
    return logits, loss


def label_smoothing_body(_, goldx, goldy, features, labels, mode, params):
    logits = model(features, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            smoothed_smax_xent_loss(output=logits, targets=labels, smooth_val=params.get("smooth_val", 0.1)))
    else:
        loss = None
    return logits, loss


def convex_label_smoothing_body(_, goldx, goldy, features, labels, mode, params):
    logits = model(features, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            convex_smoothed_smax_xent_loss(output=logits, targets=labels, smooth_val=params.get("smooth_val", 0.1)))
    else:
        loss = None
    return logits, loss


def convex_label_smoothing_body_hard(_, goldx, goldy, features, labels, mode, params):
    logits = model(features, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            convex_smoothed_smax_xent_loss(output=logits, targets=labels, smooth_val=params.get("smooth_val", 0.1)))
    else:
        loss = None
    return logits, loss


def convex_label_smoothing_body_hard_scheduled(self, goldx, goldy, features, labels, mode, params):
    logits = model(features, n_classes=params["n_class"], dropout=mode == tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        smooth_val = self.smooth_val * (
            tf.minimum(tf.train.get_or_create_global_step() / params["n_steps"], 1.0))
        smooth_val = tf.to_float(smooth_val)
        loss = tf.reduce_mean(
            convex_smoothed_smax_xent_loss(output=logits, targets=labels, smooth_val=smooth_val))
    else:
        loss = None
    return logits, loss


def smax_xent_loss(output, targets, n_class=None):
    if n_class is None:
        n_class = output.get_shape().as_list()[-1]
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(targets, depth=n_class),
                                                      logits=output)


def smoothed_smax_xent_loss(output, targets, smooth_val=0.1):
    n_classes = tf.shape(output)[-1]
    smooth_labels = tf.one_hot(targets, depth=n_classes, dtype=tf.float32) * (
            1.0 - smooth_val) + smooth_val / tf.to_float(n_classes)
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=smooth_labels, logits=output)


def convex_smoothed_smax_xent_loss(output, targets, smooth_val=0.1):
    n_classes = tf.shape(output)[-1]
    smooth_labels = tf.one_hot(targets, depth=n_classes, dtype=tf.float32) * (
            1.0 - smooth_val) + smooth_val * tf.stop_gradient(
        tf.nn.softmax(output))
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=smooth_labels, logits=output)


def convex_smoothed_smax_xent_loss_hard(output, targets, smooth_val=0.1):
    n_classes = tf.shape(output)[-1]
    smooth_labels = tf.one_hot(targets, depth=n_classes, dtype=tf.float32) * (
            1.0 - smooth_val) + smooth_val * tf.stop_gradient(
        tf.one_hot(tf.argmax(output), depth=n_classes))
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=smooth_labels, logits=output)


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer"),
    ("Resampler", "CorruptiveOversamplerWithOriginalTargs")])
class RelabelHighLossExamples(TFEstimatorExperiment):
    estimator = baseline_body

    def fit(self, X, y):
        y_true = [x["y"] for x in X]
        X = [x["X"] for x in X]
        super().fit(X, y)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X, dtype=np.float32), num_epochs=1,
                                                              shuffle=False)
        preds = self.model.predict(input_fn=predict_input_fn)
        prob_per_example = [p["prob"] for p in preds]
        neg_log_likelihoods = [-np.log(p[i]) for p, i in zip(prob_per_example, self.label_enc.transform(y))]
        top_10_percent = sorted(range(len(y)), key=lambda i: neg_log_likelihoods[i], reverse=True)[:len(y) // 10]
        # relabel top 10%
        new_y = [y_noisy if i not in top_10_percent else y_real for i, (y_noisy, y_real) in enumerate(zip(y, y_true))]
        new_y = self.label_enc.transform(new_y)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X, np.float32), y=np.asarray(new_y),
                                                            num_epochs=None, shuffle=True)
        self.model.train(input_fn=train_input_fn, steps=5000)


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer"),
    ("Resampler", "CorruptiveOversamplerWithOriginalTargs")])
class RelabelLowLossExamples(TFEstimatorExperiment):
    estimator = baseline_body

    def fit(self, X, y):
        y_true = [x["y"] for x in X]
        X = [x["X"] for x in X]
        super().fit(X, y)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X, dtype=np.float32), num_epochs=1,
                                                              shuffle=False)
        preds = self.model.predict(input_fn=predict_input_fn)
        prob_per_example = [p["prob"] for p in preds]
        neg_log_likelihoods = [-np.log(p[i]) for p, i in zip(prob_per_example, self.label_enc.transform(y))]
        top_10_percent = sorted(range(len(y)), key=lambda i: neg_log_likelihoods[i], reverse=False)[:len(y) // 10]
        # relabel top 10%
        new_y = [y_noisy if i not in top_10_percent else y_real for i, (y_noisy, y_real) in enumerate(zip(y, y_true))]
        new_y = self.label_enc.transform(new_y)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x=np.asarray(X, np.float32), y=np.asarray(new_y),
                                                            num_epochs=None, shuffle=True)
        self.model.train(input_fn=train_input_fn, steps=5000)


from sklearn.svm import SVC

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer"),
    ("Resampler", "CorruptiveResamplerWithOriginalTargs")])
class SVMBaseline(ClassificationExperiment):
    """
    Base class for classification models that select hyperparameters via cross validation.
    Assumes the `base_model` property set on child classes inherits from
    `sklearn.base.BaseEstimator` and implements `predict_proba` and `score`.

    :cvar base_model: Class name of base model, must be set by child classes.
    :cvar param_grid: Dictionary that maps from class paramaters to an array of values to search over.  Must be set by child classes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(auto_resample=False, *args, **kwargs)
        self.best_model = None
        self.base_model = SVC
        self.param_grid = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100], 'probability' :[True]}

    def fit(self, X, y):
        """
        Runs grid search over `self.param_grid` on `self.base_model` to optimize hyper-parameters using
        KFolds cross-validation, then retrains using the selected parameters on the full training set.

        :param X: `np.ndarray` of input features sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        X = [x["X"] for x in X]

        classifier = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid,
            cv=OversampledKFold(self.resampler_)
        )
        classifier.fit(X, y)
        self.best_model = self.base_model(**classifier.best_params_)
        self.best_model.fit(X, y)

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.best_model.classes_
        probabilities = self.best_model.predict_proba(X)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})

@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer"),
    ("Resampler", "CorruptiveResamplerWithOriginalTargs")])
class SVMWithBalance(SVMBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_grid = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100], 'probability' :[True], "class_weight": ["balanced"]}


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer"),
    ("Resampler", "CorruptiveResamplerWithOriginalTargs")])
class RelabelSupportVectors(ClassificationExperiment):
    """
    Base class for classification models that select hyperparameters via cross validation.
    Assumes the `base_model` property set on child classes inherits from
    `sklearn.base.BaseEstimator` and implements `predict_proba` and `score`.

    :cvar base_model: Class name of base model, must be set by child classes.
    :cvar param_grid: Dictionary that maps from class paramaters to an array of values to search over.  Must be set by child classes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(auto_resample=False, *args, **kwargs)
        self.best_model = None
        self.base_model = SVC
        self.param_grid = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100], 'probability' :[True]}

    def fit(self, X, y):
        """
        Runs grid search over `self.param_grid` on `self.base_model` to optimize hyper-parameters using
        KFolds cross-validation, then retrains using the selected parameters on the full training set.

        :param X: `np.ndarray` of input features sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        y_true = [x["y"] for x in X]
        X = [x["X"] for x in X]

        classifier = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid,
            cv=OversampledKFold(self.resampler_)
        )
        classifier.fit(X, y)
        self.best_model = self.base_model(**classifier.best_params_)
        self.best_model.fit(*self.resample(X, y))
        print(len(self.best_model.support_))
        top_10_percent = self.best_model.support_[:len(y) // 10]

        new_y = [y_noisy if i not in top_10_percent else y_real for i, (y_noisy, y_real) in enumerate(zip(y, y_true))]
        classifier = GridSearchCV(
            self.base_model(),
            param_grid=self.param_grid,
            cv=OversampledKFold(self.resampler_)
        )
        classifier.fit(X, y)
        self.best_model = self.base_model(**classifier.best_params_)
        self.best_model.fit(*self.resample(X, new_y))

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        labels = self.best_model.classes_
        probabilities = self.best_model.predict_proba(X)
        return pd.DataFrame({label: probabilities[:, i] for i, label in enumerate(labels)})


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class Reweighting(TFEstimatorExperiment):
    estimator = reweighting_body


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class ReweightingBaseline(TFEstimatorExperiment):
    estimator = baseline_body


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class TrainOnlyOnGoldBaseline(TFEstimatorExperiment):
    estimator = train_on_gold_body


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class LabelSmooth(TFEstimatorExperiment):
    estimator = label_smoothing_body


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class ConvexLabelSmooth(TFEstimatorExperiment):
    estimator = convex_label_smoothing_body


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class ConvexLabelSmoothHard(TFEstimatorExperiment):
    estimator = convex_label_smoothing_body_hard


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class ConvexLabelSmoothHardScheduled001(TFEstimatorExperiment):
    smooth_val = 0.01
    estimator = convex_label_smoothing_body_hard_scheduled


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class ConvexLabelSmoothHardScheduled01(TFEstimatorExperiment):
    smooth_val = 0.1
    estimator = convex_label_smoothing_body_hard_scheduled


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class ConvexLabelSmoothHardScheduled02(TFEstimatorExperiment):
    smooth_val = 0.2
    estimator = convex_label_smoothing_body_hard_scheduled


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class ConvexLabelSmoothHardScheduled05(TFEstimatorExperiment):
    smooth_val = 0.5
    estimator = convex_label_smoothing_body_hard_scheduled


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "not PlainTextFeaturizer")])
class GoldEveryBatch(TFEstimatorExperiment):
    estimator = gold_every_batch_body

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
