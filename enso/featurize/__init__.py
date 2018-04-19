"""
.. automodule:: featurize

    Entrypoint for running featurization according to config.py.
"""

import logging

import pandas as pd

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from enso.config import FEATURIZERS, DATA, N_CORES
from enso.utils import get_plugins, feature_set_location, BaseObject
from sklearn.externals import joblib


POOL = ThreadPoolExecutor(N_CORES)


class Featurization(object):
    """Class for wrapped featurization functionality."""

    def __init__(self):
        """Responsible for searching featurizer module and importing those specified in config."""
        self.featurizers = get_plugins('featurize', FEATURIZERS)

    def run(self):
        """Responsible for running actual featurization jobs."""
        futures = {}
        for featurizer in self.featurizers:
            featurizer.load()
            for dataset_name in DATA:
                dataset = self._load_dataset(dataset_name)
                logging.info("Featurizing {} with {}....".format(dataset_name, featurizer.__class__.__name__))
                future = POOL.submit(featurizer.generate, dataset, dataset_name)
                futures[future] = (featurizer, dataset_name)

        for future in concurrent.futures.as_completed(futures):
            featurizer, dataset_name = futures[future]
            try:
                future.result()
                logging.info("Completed featurization of dataset `{dataset_name}` with featurizer `{featurizer}`.".format(
                    dataset_name=dataset_name,
                    featurizer=featurizer.__class__.__name__
                ))
            except Exception as e:
                logging.exception("Failed featurization of dataset `{dataset_name}` with featurizer `{featurizer}`.".format(
                    dataset_name=dataset_name,
                    featurizer=featurizer.__class__.__name__
                ))

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

    def load(self):
        """
        Method called in flow of `python -m enso.featurize` to prevent loading
        pre-trained models into memory on file import.

        If loading a pre-trained model into memory is not required, `Featurizer.load()`
        defaults to `pass`.
        """
        pass

    def generate(self, dataset, dataset_name):
        """Responsible for generating appropriately named feature datasets."""
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
        new_dataset['Features'] = features
        self._write(new_dataset, dataset_name)

    def _write(self, featurized_dataset, dataset_name):
        """Responsible for taking a featurized dataset and writing it out to the filesystem."""
        dump_location = feature_set_location(dataset_name, self.__class__.__name__)
        joblib.dump(featurized_dataset, dump_location)
