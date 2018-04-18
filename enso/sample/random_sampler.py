from . import Sampler
import numpy as np


class Random(Sampler):
    def sample(self):
        points = self.choose_starting_points()
        return points + list(np.random.choice(self.train_indices, self.train_size - len(points), replace=False))
