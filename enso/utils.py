"""General util functions."""
import os
import os.path
from time import strptime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from enso.config import RESULTS_DIRECTORY, FEATURES_DIRECTORY


def feature_set_location(dataset_name, featurizer_name):
    """Responsible for generating filenames for generated feature sets."""
    base_dir, _, filename = dataset_name.rpartition('/')
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