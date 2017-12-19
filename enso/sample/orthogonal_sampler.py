from . import Sampler
import numpy as np


class Orthogonal(Sampler):
    DISTANCE_FUNCTION="cosine"

    def sample(self):
        centers = [self.choose_random()]
        while len(centers) < self.train_size:
            reduced = np.prod(self.distances[centers], 0)
            center = np.argmax(reduced)
            centers.append(center)
        return centers