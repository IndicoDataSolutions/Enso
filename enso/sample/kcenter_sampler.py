from . import Sampler
import numpy as np


class KCenter(Sampler):
    DISTANCE_FUNCTION = "euclidean"

    def sample(self):
        centers = self.choose_starting_points()
        while len(centers) < self.train_size:
            mins = np.min(self.distances[centers], axis=0)
            center = np.argmax(mins)
            centers.append(center)
        return centers