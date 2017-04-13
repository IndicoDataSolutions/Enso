"""Main method for running featurization according to config.py."""

import os
import inspect
from importlib import import_module

import pandas as pd

from config import FEATURIZERS, DATA


class Featurization(object):
    """Class for wrapped featurization functionality."""

    def __init__(self):
        """Responsible for searching featurizer module and importing those specified in config."""
        root, dirs, files = os.walk('Featurize').next()
        self.featurizer_classes = []
        self.featurizers = []
        for filename in files:
            if "__init__" not in filename and not filename.endswith('.pyc'):
                module_name = filename.rpartition('.')[0]
                mod = import_module("%s.%s" % (root, module_name))
                for name, obj in inspect.getmembers(mod):
                    if inspect.isclass(obj) and obj.__name__ in FEATURIZERS:
                        self.featurizer_classes.append(obj)
        names = set(item.__name__ for item in self.featurizer_classes)
        if names != FEATURIZERS:
            raise ValueError("""
                Config doesn't match featurizers present.\nFeaturizers: %s\nConfig: %s
            """ % (names, FEATURIZERS))

    def initialize(self):
        """Responsible for instantiating featurizers, taking care of any class setup."""
        for featurizer in self.featurizer_classes:
            self.featurizers.append(featurizer())

    def run(self):
        """Responsible for running actual featurization jobs."""
        for dataset_name in DATA:
            dataset = self._load_dataset(dataset_name)
            for featurizer in self.featurizers:
                featurizer.generate(dataset, dataset_name)

    @staticmethod
    def _load_dataset(dataset_name):
        """Responsible for finding datasets and reading them into dataframes."""
        df = pd.read_csv("Data/%s.csv" % dataset_name)
        if 'Text' not in df:
            raise ValueError("File: %s has no column 'Text'" % dataset_name)
        if 'Target_1' not in df:
            raise ValueError("File %s has no column 'Target_1'" % dataset_name)
        return df


if __name__ == "__main__":
    print "Featurization Started..."
    featurization = Featurization()
    print "Initializing Featurizers..."
    featurization.initialize()
    print "Converting Datasets..."
    featurization.run()
