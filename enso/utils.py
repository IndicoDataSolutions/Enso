"""General util functions."""
import os
import os.path
from time import strptime
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from enso.config import RESULTS_DIRECTORY, FEATURES_DIRECTORY
from collections import defaultdict


class SafeStratifiedShuffleSplit(StratifiedShuffleSplit):
    def _choose_starting_points(self, idxs, Y, n_points=1):
        """
        Ensures a minimum of one label per class is chosen
        """
        train_idxs = []
        test_idxs = []
        classes = set(Y)
        idxs_by_class = defaultdict(list)
        for i, y in enumerate(Y):
            idxs_by_class[y].append(i)
        for _ in range(n_points):
            for c in classes:
                train_index = random.choice(idxs_by_class[c])
                idxs_by_class[c].remove(train_index)
                train_idxs.append(train_index)
                test_index = random.choice(idxs_by_class[c])
                idxs_by_class[c].remove(test_index)
                test_idxs.append(test_index)
        remaining_idxs = [idx for idxs in idxs_by_class.values() for idx in idxs]
        return train_idxs, test_idxs, remaining_idxs

    def split(self, X, Y):
        Y = np.asarray(Y)
        idxs = np.arange(len(Y))
        for _ in range(self.n_splits):
            train_idxs, test_idxs, remaining_idxs = self._choose_starting_points(idxs, Y, n_points=1)
            try:
                train_remaining_idxs, test_remaining_idxs = train_test_split(
                    remaining_idxs, test_size=1 / self.n_splits, stratify=Y[remaining_idxs]
                )
            except ValueError:
                # In the remaning datapoints, we don't have at least 2 examples per class. Settle for random split
                # since we've already selected at least one point per class per split.
                train_remaining_idxs, test_remaining_idxs = train_test_split(
                    remaining_idxs, test_size=1 / self.n_splits
                )
            yield train_remaining_idxs + train_idxs, test_remaining_idxs + test_idxs


class RationalizedStratifiedShuffleSplit(SafeStratifiedShuffleSplit):
    def split(self, X, Y):
        yield from super().split(X, [y[1] for y in Y])

class HackSplit(object):
    def __init__(self, n_splits, test_size):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X, Y):
        Y = np.asarray(Y)
        idxs = np.arange(len(Y))
        test_idxs = idxs[-self.test_size:]
        train_idxs = idxs[:-self.test_size]
        for _ in range(self.n_splits):
            yield train_idxs, test_idxs

def feature_set_location(dataset_name, featurizer_name):
    """Responsible for generating filenames for generated feature sets."""
    base_dir, _, filename = dataset_name.rpartition("/")
    write_location = "%s/%s/" % (FEATURES_DIRECTORY, base_dir)
    dump_name = "%s_%s_features.csv" % (filename, featurizer_name)
    return write_location + dump_name


def get_all_experiment_runs():
    """Grab all experiment runs and return a list sorted by date."""
    dirs = [item for item in os.listdir(RESULTS_DIRECTORY)]
    timed_dirs = []
    for directory in dirs:
        try:
            dt = strptime(directory, "%Y-%m-%d_%H-%M-%S")
            timed_dirs.append((directory, dt))
        except ValueError:
            pass
    timed_dirs.sort(key=lambda pair: pair[1], reverse=True)
    dirs = [pair[0] for pair in timed_dirs]
    return dirs


def labels_to_binary(target_list):
    """
    Convert a list of labels into a pandas dataframe appropriate for metric calculations.

    Example: labels_to_binary('apple', ['apple', 'orange']) ->
    pd.DataFrame({'apple': [1, 0], 'orange': [0, 1]})
    """
    full_mapping = {}
    for target in set(target_list):
        full_mapping[target] = [int(item == target) for item in target_list]
    return pd.DataFrame(full_mapping)


class BaseObject(object):
    """Base object for all plugins."""

    def name(self):
        """
        Prints the name of the current class to aid logging and result formatting.
        """
        return self.__class__.__name__


class OversampledKFold(StratifiedKFold):
    def __init__(self, resampler, n_splits=3, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle, random_state)
        self.resampler = resampler

    def split(self, X, y, groups=None):
        y_ = np.asarray(y)
        for tr, te in super().split(X, y, groups):
            yield self.resampler.resample(tr, y_[tr])[0], te
