from . import Sampler
import random
import numpy as np
from collections import Counter, defaultdict
from enso.registry import ModeKeys, Registry
import itertools


@Registry.register_sampler(ModeKeys.CLASSIFY)
class ImbalanceSampler(Sampler):
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

        :returns: np.array of example indices selected by imbalance sampling
        """
        y = np.asarray(self.train_labels)
        class_counts = Counter(y)
        num_classes = len(class_counts.keys())
        desired_counts = {}
        max_count = max(class_counts.values())
        for i, (label, count) in enumerate(class_counts.most_common()):
            imbalance_fraction = 1 / (i + 1)
            desired_counts[label] = max(
                min(count, int(imbalance_fraction * max_count)), 1
            )

        idxs_by_y = defaultdict(list)
        for i, element in enumerate(y):
            idxs_by_y[element].append(i)

        idx_sample = []
        for class_name in class_counts:
            sample_idxs = np.random.choice(
                idxs_by_y[class_name], desired_counts[class_name], replace=False
            ).tolist()
            idx_sample.extend(sample_idxs)

        random.shuffle(idx_sample)
        assert len(idx_sample) == np.sum(list(desired_counts.values()))
        return idx_sample
