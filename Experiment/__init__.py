"""Module for all experiment methods that one might want to test."""
import abc
from functools import wraps

import numpy as np

from utils import BaseObject


class CheckOutput(abc.ABCMeta):
    """Decorator class to add output checks to different experiments classes."""

    def __new__(cls, *args):
        """Wrap predict method with appropriate decorator check."""
        experiment_class = super(CheckOutput, cls).__new__(cls, *args)
        experiment_class.predict = experiment_class.verify_output(experiment_class.predict)
        return experiment_class


class Experiment(BaseObject):
    """Base class for all Experiments."""

    __metaclass__ = CheckOutput

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

    @classmethod
    def verify_output(cls, function):
        """Check predict output to ensure it complies with the experiment type."""
        raise NotImplementedError


class ClassificationExperiment(Experiment):
    """Base class for classification experiments."""

    @classmethod
    def verify_output(cls, func):
        """Verify the output of classification tasks."""
        @wraps(func)
        def wrapped_predict(self, dataset):
            response = func(dataset)
            # Must have a prediction for each entry
            assert len(response) == len(dataset)
            # All probabilities should sum to approx. 1
            assert np.all(np.isclose(response.sum(axis=1), np.ones(len(response))))
            return response
        return func


class RegressionExperiment(Experiment):
    """Base class for regression experiments."""

    pass


class MatchingExperiment(Experiment):
    """Base class for matching experiments."""

    pass
