"""Basic classification metrics."""
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import numpy as np

from enso.metrics import ClassificationMetric
from enso.utils import labels_to_binary


class WeightedRocAuc(ClassificationMetric):
    """Classwise ROC AUC metric."""

    def evaluate(self, ground_truth, result):
        """Return AUC metric."""
        classwise_auc = {}
        binary_labels = labels_to_binary(ground_truth)
        binary_labels = np.hstack([binary_labels[column].values.reshape(-1, 1) for column in result.columns])
        predicted_labels = np.hstack([result[column].values.reshape(-1, 1) for column in result.columns])
        return roc_auc_score(binary_labels, predicted_labels, average='weighted')


class MacroRocAuc(ClassificationMetric):
    """Classwise ROC AUC metric."""

    def evaluate(self, ground_truth, result):
        """Return AUC metric."""
        classwise_auc = {}
        binary_labels = labels_to_binary(ground_truth)
        binary_labels = np.hstack([binary_labels[column].values.reshape(-1, 1) for column in result.columns])
        predicted_labels = np.hstack([result[column].values.reshape(-1, 1) for column in result.columns])
        return roc_auc_score(binary_labels, predicted_labels, average='macro')


class Accuracy(ClassificationMetric):
    """Return class accuracy"""
    def evaluate(self, ground_truth, result):
        predicted = result.idxmax(axis=1)
        return accuracy_score(ground_truth, predicted)


class LogLoss(ClassificationMetric):
    """LogLoss"""

    def evaluate(self, ground_truth, result):
        """Return AUC metric."""
        binary_labels = labels_to_binary(ground_truth)
        binary_labels = np.hstack([binary_labels[column].values.reshape(-1, 1) for column in result.columns])
        predicted_labels = np.hstack([result[column].values.reshape(-1, 1) for column in result.columns])
        return log_loss(binary_labels, predicted_labels)



