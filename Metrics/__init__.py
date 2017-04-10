"""Module for all metrics that one might want for evaluation."""


class Metric(object):
    """Base class for all Metrics."""

    pass


class ClassificationMetric(Metric):
    """Base class for classification metrics."""

    pass


class RegressionMetric(Metric):
    """Base class for regression metrics."""

    pass


class MatchingMetric(Metric):
    """Base class for matching metrics."""

    pass
