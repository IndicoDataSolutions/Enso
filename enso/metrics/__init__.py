"""Module for all metrics that one might want for evaluation."""
import abc

from enso.utils import BaseObject


class Metric(BaseObject):
    """Base class for all Metrics."""

    @abc.abstractmethod
    def evaluate(self, ground_truth, result):
        """
        General metric endpoint for generating results.

        This should be implemented by every metric class. No exceptions.
        """
        raise NotImplementedError


class ClassificationMetric(Metric):
    """Base class for classification metrics."""


class RegressionMetric(Metric):
    """Base class for regression metrics."""


class MatchingMetric(Metric):
    """Base class for matching metrics."""


class SequenceLabelingMetric(Metric):
    """ Base class for Sequence Labeling metrics"""


from enso.metrics import basic_classification
from enso.metrics import basic_seq_labeling
