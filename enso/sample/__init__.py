"""
Base class definition for all training data samples.
Samplers aid transfer learning methods by attempting to request
labeled examples that are more informative than a simple random sample of training data
"""

import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances
from enso.registry import Registry


def sample(sampler, data, train_labels, train_indices, train_size):
    sampler = Registry.get_sampler(sampler)(data, train_labels, train_indices, train_size)
    return sampler.sample()


class Sampler:
    """
    Base class for all `Sampler`'s

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """

    def __init__(self, data, train_labels, train_indices, train_size):
        """
        Given a set of training data and a desired size, uses a strategy to
        select a subset of datapoints to use as labeled training data.
        All arguments are taken at initialization.  The :func:`sample` function
        may be called multiple times on a single `Sampler` object.


        """
        self.data = data
        self.train_labels = train_labels
        self.train_indices = train_indices
        self.train_size = train_size
        if len(self.classes) > train_size:
            raise ValueError("The train size can not be smaller than the number of classes.")

    @property
    def classes(self):
        if not hasattr(self, '_classes'):
            self._classes = set(self.train_labels)
        return self._classes

    def _choose_starting_points(self):
        """
        Ensures a minimum of one label per class is chosen
        """
        points = []
        for cls in self.classes:
            indices = [i for i, val in enumerate(self.train_labels) if val == cls]
            index = random.choice(indices)
            points.append(self.train_indices[index])
        return points

    @property
    def distances(self):
        if not hasattr(self, '_distances'):
            self._distances = pairwise_distances(self.points, metric=self.DISTANCE_FUNCTION)
        return self._distances

    @property
    def points(self):
        return [np.array(point) for point in self.data]

    def sample(self):
        """
        Given the settings provided at initialization, apply the provided sampling strategy

        :returns: np.array of example indices selected by `Sampler`.
        """
        raise NotImplementedError


from enso.sample import kcenter_sampler
from enso.sample import no_sampler
from enso.sample import orthogonal_sampler
from enso.sample import random_sampler
