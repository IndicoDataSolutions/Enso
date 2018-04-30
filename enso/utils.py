"""General util functions."""
import os
import os.path
import inspect
from time import strptime
from importlib import import_module

import pandas as pd

from enso.config import RESULTS_DIRECTORY, FEATURES_DIRECTORY


def feature_set_location(dataset_name, featurizer_name):
    """Responsible for generating filenames for generated feature sets."""
    base_dir, _, filename = dataset_name.rpartition('/')
    write_location = "%s/%s/" % (FEATURES_DIRECTORY, base_dir)
    dump_name = "%s_%s_features.csv" % (filename, featurizer_name)
    return write_location + dump_name


def get_plugins(plugin_dir, match_names, return_class=False):
    """Responsible for grabbing objects from specified plugin directories."""
    full_plugin_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), plugin_dir))
    root, dirs, files = next(os.walk(full_plugin_dir_path))

    relevant_classes = []
    for filename in files:
        if not filename.endswith('.py') or '__' in filename:
            continue
        module_name = filename.rpartition('.')[0]
        import_path = "%s.%s.%s" % ("enso", plugin_dir, module_name)
        mod = import_module(import_path)
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and obj.__name__ in match_names:
                relevant_classes.append(obj)

    names = set(item.__name__ for item in relevant_classes)
    if names != match_names:
        raise ValueError("""
            Config doesn't match classes present.\n%s: %s\nConfig: %s
        """ % (plugin_dir, names, match_names))

    if return_class:
        return relevant_classes
    else:
        return [relevant_class() for relevant_class in relevant_classes]


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
