"""Module for all experiment methods that one might want to test."""
import abc

from utils import BaseObject


class Experiment(BaseObject):
    """Base class for all Experiments."""

    @abc.abstractmethod
    def train(self, dataset, target):
        """
        General endpoint to run training for a given experiment.

        dataset is a sub-selected version of the dataset to avoid test/train contamination.
        Target refers to the column of interest.

        Even if this method doesn't do anything it must be implemented
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, dataset):
        """
        General endpoint to predict on the test set for a given experiment.

        Dataset only contains relvant rows. Prediction format is dependant on class of experiment
        """
        raise NotImplementedError


class ClassificationExperiment(Experiment):
    """Base class for classification experiments."""

    pass


class RegressionExperiment(Experiment):
    """Base class for regression experiments."""

    pass


class MatchingExperiment(Experiment):
    """Base class for matching experiments."""

    pass
