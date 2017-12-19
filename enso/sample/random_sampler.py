from . import Sampler
import numpy as np


class Random(Sampler):
    def sample(self):
        return np.random.choice(self.train_indices, self.train_size, replace=False)