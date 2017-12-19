from . import Sampler
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class KCenter(Sampler):
    DISTANCE_FUNCTION="euclidean"

    def sample(self):
        centers = [self.choose_random()]
        while len(centers) < self.train_size:
            mins = np.min(self.distances[centers], axis=0)
            center = np.argmax(mins)
            centers.append(center)
        return centers
    
    @property
    def distances(self):
        if not hasattr(self, '_distances'):
            self._distances = pairwise_distances(self.points)
        return self._distances