"""Basic classification metrics."""
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import numpy as np

from enso.metrics import ClassificationMetric
from enso.utils import labels_to_binary
from enso.registry import Registry, ModeKeys

@Registry.register_metric(ModeKeys.RATIONALIZED)
class MacroRocAucRationalized(ClassificationMetric):
    """Classwise ROC AUC metric."""

    def evaluate(self, ground_truth, result):
        """Return AUC metric."""
        ground_truth = [yi[1] for yi in ground_truth]
        classes = list(set(ground_truth) | set(result.columns))
        classwise_auc = {}
        binary_labels = labels_to_binary(ground_truth)
        default = np.zeros(shape=[len(ground_truth), 1], dtype=np.int32)
        binary_labels = np.hstack([binary_labels[column].values.reshape(-1, 1) if column in binary_labels else default for column in classes])
        predicted_labels = np.hstack([result[column].values.reshape(-1, 1) if column in result.columns else default for column in classes])
        print(binary_labels[:10])
        print(predicted_labels[:10])
        return roc_auc_score(binary_labels, predicted_labels, average='macro')

@Registry.register_metric(ModeKeys.RATIONALIZED)
class AccuracyRationalized(ClassificationMetric):
    """Return class accuracy"""
    def evaluate(self, ground_truth, result):
        ground_truth = [yi[1] for yi in ground_truth]
        predicted = result.idxmax(axis=1)
        return accuracy_score(ground_truth, predicted)
