"""Module for all metrics that one might want for evaluation."""
import abc

from enso.utils import BaseObject
from enso.config import MODE


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

    def __new__(cls):
        if MODE != "Classification":
            raise ValueError("Incorrect Metrics for {} task".format(MODE))
        return super().__new__(cls)


class RegressionMetric(Metric):
    """Base class for regression metrics."""

    pass


class MatchingMetric(Metric):
    """Base class for matching metrics."""

    pass


class SequenceLabelingMetric(Metric):
    """ Base class for Sequence Labeling metrics"""
    def __new__(cls):
        if MODE != "SequenceLabeling":
            raise ValueError("Incorrect Metrics for {} task".format(MODE))
        return super().__new__(cls)

