from . import Sampler
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
        points = self._choose_starting_points(n_points=3)
        train_indices = [idx for idx in self.train_indices if idx not in points]
        points += list(np.random.choice(train_indices, self.train_size - len(points), replace=False))
        return points



@Registry.register_sampler(ModeKeys.RATIONALIZED)
class RandomRationalized(Random):
    """
    Randomly selects examples from the training dataset.

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """
    @property
    def classes(self):
        if not hasattr(self, "_classes"):
            self._classes = set([label[1] for label in self.train_labels])
        return self._classes

    def _choose_starting_points(self, n_points=1):
        """
        Ensures a minimum of one label per class is chosen
        """
        points = []
        for _ in range(n_points):
            for cls in self.classes:
                indices = [i for i, val in enumerate(self.train_labels) if val[1] == cls]
                index = random.choice(indices)
                points.append(self.train_indices[index])
        return points



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

    def _choose_starting_points(self, n_points=3):
        """
        Ensures a minimum of one label per class is chosen
        """
        points = []
        for _ in range(n_points):
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




