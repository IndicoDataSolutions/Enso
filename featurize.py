"""Main method for running featurization according to config.py."""
import logging

import pandas as pd

from config import FEATURIZERS, DATA
from utils import get_plugins


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
    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing Featurizers...")
    featurization = Featurization()
    logging.info("Converting Datasets...")
    featurization.run()
