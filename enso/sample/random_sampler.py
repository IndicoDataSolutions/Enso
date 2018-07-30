from . import Sampler
from enso.config import MODE
import random
import numpy as np
from enso.registry import ModeKeys, Registry
import itertools


@Registry.register_sampler(ModeKeys.CLASSIFY)
class Random(Sampler):
    """
    Randomly selects examples from the training dataset.

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """

    def sample(self):
        """
        Randomly samples feature vectors.

        :returns: np.array of example indices selected by random sampling
        """
        points = self._choose_starting_points()
        return points + list(np.random.choice(self.train_indices, self.train_size - len(points), replace=False))


@Registry.register_sampler(ModeKeys.SEQUENCE)
class RandomSequence(Random):
    """
    Random Sampler, but for sequence labelling tasks.

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """

    def __init__(self, data, train_labels, train_indices, train_size):

        self.data = data

        stripped_labels = []
        for item in train_labels:
            per_sample = []
            for label in item:
                per_sample.append(label["label"])
            stripped_labels.append(per_sample)

        self.train_labels = stripped_labels
        self.train_indices = train_indices
        self.train_size = train_size

    @property
    def classes(self):
        if not hasattr(self, '_classes'):
            self._classes = set(itertools.chain.from_iterable(self.train_labels))
        return self._classes

    def _choose_starting_points(self):
        """
        Ensures a minimum of one label per class is chosen
        """
        points = []
        for cls in self.classes:
            indices = [i for i, val in enumerate(self.train_labels) if cls in val]
            index = random.choice(indices)
            points.append(self.train_indices[index])
        return points

    @property
    def distances(self):
        raise NotImplementedError

    @property
    def points(self):
        raise NotImplementedError




