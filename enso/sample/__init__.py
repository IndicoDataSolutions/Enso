import numpy
import logging
from sklearn.metrics.pairwise import pairwise_distances

class Sampler(object):
    """Base class for all samples."""

    def __init__(self, data, train_indices, train_size):
        self.data = data
        self.train_indices = train_indices
        self.train_size = train_size

    def sample(self):
        raise NotImplementedError

    @classmethod
    def sample(cls, sampler, data, train_indices, train_size):
        return cls._class_for(sampler)(data, train_indices, train_size).sample()
    
    @classmethod
    def _class_for(cls, sampler_string):
        for subclass in cls.__subclasses__():
            if subclass.__name__ == sampler_string:
                return subclass
        raise ValueError("Invalid sampler attempted ({})".format(sampler_string))


class Random(Sampler):
    def sample(self):
        return numpy.random.choice(self.train_indices, self.train_size, replace=False)


class PointWithDistance(object):
    def __init__(self, index, distance):
        self.index = index
        self.distance = distance


class KCenter(Sampler):
    @property
    def points(self):
        return [numpy.array(point) for point in self.data]

    def sample(self):
        logging.info("Sampling for KCenter")
        random_center = numpy.random.choice(self.train_indices)
        centers = [random_center]
        while len(centers) < self.train_size:
            min_distances = []
            for index, point in enumerate(self.points):
                distances = []
                if index in centers:
                    continue
                distances = []
                for center in centers:
                    point_with_distance = PointWithDistance(index, self.distances[index][center])
                    distances.append(point_with_distance)
                new_min_point = min(distances, key=lambda point_with_dist: point_with_dist.distance)
                min_distances.append(new_min_point)
            new_center_point = max(min_distances,  key=lambda point_with_dist: point_with_dist.distance)
            centers.append(new_center_point.index)
            logging.info("Left to get {} for train size of {}".format(str(self.train_size - len(centers)), str(self.train_size)))
        return centers

    @property
    def distances(self):
        if not hasattr(self, '_distances'):
            self._distances = pairwise_distances(self.points)
        return self._distances