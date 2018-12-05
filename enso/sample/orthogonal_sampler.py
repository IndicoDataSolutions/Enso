from . import Sampler
import numpy as np

from enso.registry import Registry, ModeKeys


@Registry.register_sampler(ModeKeys.CLASSIFY)
class Orthogonal(Sampler):
    """
    Randomly selects starting points, then selects additional points for which
    the product of the cosine distance to all starting points is maximally large.
    Each selected point is then iteratively added to the list of starting points.

    :param data: pd.Series of feature vectors
    :param train_labels: pd.Series of targets
    :param train_indices: pd.Series of example indices
    :param train_size: int number of examples to select
    """
    DISTANCE_FUNCTION = "cosine"

    def sample(self):
        """
        Applies the orthogonal sampling strategy.

        :returns: np.array of example indices selected by `Orthogonal`.
        """
        centers = self._choose_starting_points()
        while len(centers) < self.train_size:
            reduced = np.prod(self.distances[centers], 0)
            center = np.argmax(np.absolute(reduced))
            centers.append(center)
        return centers