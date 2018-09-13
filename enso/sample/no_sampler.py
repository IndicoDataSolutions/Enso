from . import Sampler
import numpy as np
from enso.registry import ModeKeys, Registry


@Registry.register_sampler(ModeKeys.ANY)
class NoSampler(Sampler):
    """
    Randomly selects examples from the training dataset.

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """

    def __init__(self, data, train_labels, train_indices, train_size):
        self.train_size = train_size
        self.train_indices = train_indices

    def sample(self):
        idx_copy = np.copy(self.train_indices)
        np.random.shuffle(self.train_indices)
        return idx_copy[:self.train_size]
