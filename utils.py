"""General util functions."""
import os
import inspect
from time import strptime
from importlib import import_module

import pandas as pd


def feature_set_location(dataset_name, featurizer):
    """Responsible for generating filenames for generated feature sets."""
    base_dir, _, filename = dataset_name.rpartition('/')
    write_location = "features/%s/" % base_dir
    dump_name = "%s_%s_features.csv" % (filename, featurizer.__class__.__name__)
    return write_location + dump_name


def get_plugins(plugin_dir, match_names):
    """Responsible for grabbing objects from specified plugin directories."""
    root, dirs, files = os.walk(plugin_dir).next()
    relevant_classes = []
    for filename in files:
        if "__init__" not in filename and not filename.endswith('.pyc'):
            module_name = filename.rpartition('.')[0]
            mod = import_module("%s.%s" % (root, module_name))
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and obj.__name__ in match_names:
                    relevant_classes.append(obj)
    names = set(item.__name__ for item in relevant_classes)
    if names != match_names:
        raise ValueError("""
            Config doesn't match classes present.\n%s: %s\nConfig: %s
        """ % (plugin_dir, names, match_names))

    return [relevant_class() for relevant_class in relevant_classes]


def get_all_experiment_runs():
    """Grab all experiment runs and return a list sorted by date."""
    root, dirs, files = os.walk('results').next()
    dirs.sort(key=lambda d: strptime(d, "%Y-%m-%d_%H-%M-%S"), reverse=True)
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
        """Helper function to grab class names cleanly."""
        return self.__class__.__name__
