from . import Sampler
import numpy as np


class Random(Sampler):
    """
    Randomly selects examples from the training dataset.

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """

    def sample(self):
        """
        Randomly samples feature vectors.

        :returns: np.array of example indices selected by random sampling
        """
        points = self._choose_starting_points()
        return points + list(np.random.choice(self.train_indices, self.train_size - len(points), replace=False))
