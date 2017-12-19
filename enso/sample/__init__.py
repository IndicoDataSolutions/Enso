import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from enso.config import TEST_SETUP


def sample(sampler, data, train_labels, train_indices, train_size):
    sampler = Sampler._class_for(sampler)(data, train_labels, train_indices, train_size)
    return sampler.sample()


class Sampler(object):
    """Base class for all samples."""

    def __init__(self, data, train_labels, train_indices, train_size):
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

    def choose_starting_points(self):
        points = []
        for klass in self.classes:
            index = self.train_labels.index(klass)
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
        raise NotImplementedError

    @classmethod
    def _class_for(cls, sampler_string):
        from enso.sample.kcenter_sampler import KCenter
        from enso.sample.random_sampler import Random
        from enso.sample.orthogonal_sampler import Orthogonal
        for subclass in cls.__subclasses__():
            if subclass.__name__ == sampler_string:
                return subclass
        raise ValueError("Invalid sampler attempted ({})".format(sampler_string))