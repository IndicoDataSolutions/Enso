"""General util functions."""
import os
import inspect
from importlib import import_module


def feature_set_location(dataset_name, featurizer):
    """Responsible for generating filenames for generated feature sets."""
    base_dir, _, filename = dataset_name.rpartition('/')
    write_location = "Features/%s/" % base_dir
    dump_name = "%s_%s_features.csv" % (filename, featurizer.__class__.__name__)
    return write_location + dump_name


def get_plugins(plugin_dir, match_names, instantiate=True):
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

    if instantiate:
        return [relevant_class() for relevant_class in relevant_classes]
    return relevant_classes


class BaseObject(object):
    """Base object for all plugins."""

    def name(self):
        """Helper function to grab class names cleanly."""
        return self.__class__.__name__
