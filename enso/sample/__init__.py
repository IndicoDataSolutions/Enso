import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def sample(sampler, data, train_indices, train_size):
    sampler = Sampler._class_for(sampler)(data, train_indices, train_size)
    return sampler.sample()


class Sampler(object):
    """Base class for all samples."""

    def __init__(self, data, train_indices, train_size):
        self.data = data
        self.train_indices = train_indices
        self.train_size = train_size

    def choose_random(self):
        return np.random.choice(self.train_indices)

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