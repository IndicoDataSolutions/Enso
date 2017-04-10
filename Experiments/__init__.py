"""Module for all experiment methods that one might want to test."""


class Experiment(object):
    """Base class for all Experiments."""

    pass


class ClassificationExperiment(Experiment):
    """Base class for classification experiments."""

    pass


class RegressionExperiment(Experiment):
    """Base class for regression experiments."""

    pass


class MatchingExperiment(Experiment):
    """Base class for matching experiments."""

    pass
