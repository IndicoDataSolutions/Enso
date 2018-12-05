from . import Sampler
import numpy as np

from enso.registry import Registry, ModeKeys


@Registry.register_sampler(ModeKeys.CLASSIFY)
class KCenter(Sampler):
    """
    Randomly selects an example from each class to use as "centers",
    then selects points that are maximally distant from any given center
    to use as training examples.

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """
    DISTANCE_FUNCTION = "euclidean"

    def sample(self):
        """
        Applies the KCenter sampling strategy.

        :returns: np.array of example indices selected by `KCenter`.
        """
        centers = self._choose_starting_points()
        while len(centers) < self.train_size:
            mins = np.min(self.distances[centers], axis=0)
            center = np.argmax(mins)
            centers.append(center)
        return centers
