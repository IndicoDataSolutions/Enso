"""
Entrypoint for featurizing datasets according to the specifications of `config.py`.
"""

import logging

import pandas as pd

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enso.config import FEATURIZERS, DATA, N_CORES
from enso.utils import get_plugins, feature_set_location, BaseObject
from sklearn.externals import joblib


class Featurization(object):
    """
    Orchestrates the application of the selected featurizers to a set of featurizers
    according to the settings specified in `config.py`
    """

    def __init__(self):
        """
        Responsible for searching featurizer module and importing those specified in config.
        """
        self.featurizers = get_plugins('featurize', FEATURIZERS)

    def _run(self, POOL):
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

    def run(self, n_jobs=N_CORES):
        """
        Responsible for ensuring every active featurizer is applied to every active dataset.
        Featurization is parallelized by default, with `n_jobs` set to number of cores specified in
        config.py.  If n_jobs is set to 1 or less, featurization is run in a single threaded setting.
        """
        if n_jobs > 1:
            POOL = ProcessPoolExecutor(n_jobs)
            self._run(POOL)
        else:
            self._run(ThreadPoolExecutor(1))

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
    """
    Base class for building featurizers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self):
        """
        Method called in flow of `python -m enso.featurize` to prevent loading
        pre-trained models into memory on file import.

        If loading a pre-trained model into memory is not required, `Featurizer.load()`
        defaults to `pass`.
        """
        pass

    def generate(self, dataset, dataset_name):
        """
        Given a `dataset` pandas DataFrame and string `dataset_name`,
        add column `"Features"` to the provided `pd.DataFrame` and serialize the result
        to the results folder listed in `config.py`.

        If a given featurizer exposes a :func:`featurize_batch` method, that method will be
        called to perform featurization.  Otherwise, :class:`Featurizer`'s will fall
        back to calling :func:`featurize` on each individual example.

        :param dataset: `pd.DataFrame` object that must contain a `Text` column.
        :param dataset_name: `str` name to use as a save location in the `config.FEATURES_DIRECTORY`.
        """
        features = []
        if callable(getattr(self, "featurize_batch", None)):
            features = self.featurize_batch(dataset['Text'])
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

    def featurize_batch(self, X, batch_size=32):
        """
        :param X: `pd.Series` that contains raw text to featurize
        :param batch_size: int number of examples to process per batch
        :returns: list of np.ndarray representations of text
        """
        raise NotImplementedError

    def featurize(self, text):
        """
        :param text: text of a singular example
        :returns: `np.ndarray` representation of text
        """
        raise NotImplementedError
