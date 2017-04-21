"""Basic classification metrics."""
from sklearn.metrics import roc_auc_score

from metrics import ClassificationMetric
from utils import labels_to_binary


class RocAuc(ClassificationMetric):
    """Classwise ROC AUC metric."""

    def evaluate(self, ground_truth, result):
        """Return AUC metric."""
        classwise_auc = {}
        binary_labels = labels_to_binary(ground_truth)
        for column in result.columns.values:
            classwise_auc[column] = roc_auc_score(binary_labels[column], result[column])
        return classwise_auc
