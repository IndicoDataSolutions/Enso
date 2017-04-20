"""Module for all metrics that one might want for evaluation."""
import abc

from utils import BaseObject


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

    pass


class RegressionMetric(Metric):
    """Base class for regression metrics."""

    pass


class MatchingMetric(Metric):
    """Base class for matching metrics."""

    pass
