import numpy
import logging
from sklearn.metrics.pairwise import pairwise_distances


def sample(sampler, data, train_indices, train_size):
    return Sampler._class_for(sampler)(data, train_indices, train_size).sample()


class Sampler(object):
    """Base class for all samples."""

    def __init__(self, data, train_indices, train_size):
        self.data = data
        self.train_indices = train_indices
        self.train_size = train_size

    def sample(self):
        raise NotImplementedError

    @classmethod
    def _class_for(cls, sampler_string):
        for subclass in cls.__subclasses__():
            if subclass.__name__ == sampler_string:
                return subclass
        raise ValueError("Invalid sampler attempted ({})".format(sampler_string))


class Random(Sampler):
    def sample(self):
        return numpy.random.choice(self.train_indices, self.train_size, replace=False)


class KCenter(Sampler):
    @property
    def points(self):
        return [numpy.array(point) for point in self.data]

    def sample(self):
        random_center = numpy.random.choice(self.train_indices)
        centers = [random_center]
        while len(centers) < self.train_size:
            mins = numpy.min(self.distances[centers], axis=0)
            center = numpy.argmax(mins)
            centers.append(center)
        return centers
    
    @property
    def distances(self):
        if not hasattr(self, '_distances'):
            self._distances = pairwise_distances(self.points)
        return self._distances