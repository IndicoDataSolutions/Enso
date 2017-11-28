"""Main method for running featurization according to config.py."""
import logging

import pandas as pd

from enso.config import FEATURIZERS, DATA
from enso.utils import get_plugins, feature_set_location, BaseObject


class Featurization(object):
    """Class for wrapped featurization functionality."""

    def __init__(self):
        """Responsible for searching featurizer module and importing those specified in config."""
        self.featurizers = get_plugins('featurize', FEATURIZERS)

    def run(self):
        """Responsible for running actual featurization jobs."""
        for dataset_name in DATA:
            dataset = self._load_dataset(dataset_name)
            for featurizer in self.featurizers:
                print("Featurizing {} with {}....".format(dataset_name, featurizer.__class__.__name__))
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


class Featurizer(BaseObject):
    """Base class for building featurizers."""

    def generate(self, dataset, dataset_name):
        """Responsible for generating appropriatelynamed feature datasets."""
        features = []
        if callable(getattr(self, "featurize_list", None)):
            features = self.featurize_list(dataset['Text'])
        elif callable(getattr(self, "featurize", None)):
            features = [self.featurize(entry) for entry in dataset['Text']]
        else:
            raise NotImplementedError("""
                Featurizers must implement the featurize_list, or the featurize method
            """)
        new_dataset = dataset.copy()  # Don't want to modify the underlying dataframe
        new_dataset['Text'] = features
        new_dataset.rename(columns={'Text': 'Features'}, inplace=True)
        self._write(new_dataset, dataset_name)

    def _write(self, featurized_dataset, dataset_name):
        """Responsible for taking a featurized dataset and writing it out to the filesystem."""
        dump_location = feature_set_location(dataset_name, self)
        featurized_dataset.to_csv(dump_location)
