"""Main method for running experiments as configured in config.py."""
import os
import json
import logging
from shutil import copyfile
from collections import defaultdict
from time import gmtime, strftime
from ast import literal_eval

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from utils import feature_set_location, get_plugins
from config import FEATURIZERS, DATA, EXPERIMENTS, METRICS, TEST_SETUP


class Experimentation(object):
    """Responsible for running experiments configured in config."""

    def __init__(self):
        """Responsible for gathering and instantiating experiments, featurizers, and metrics."""
        self.experiments = get_plugins("experiment", EXPERIMENTS)
        self.featurizers = get_plugins("featurize", FEATURIZERS)
        self.metrics = get_plugins("metrics", METRICS)

    def run_experiments(self):
        """Responsible for actually running experiments."""
        experiment_results = defaultdict(lambda: defaultdict(dict))
        for dataset_name in DATA:
            logging.info("Experimenting on %s dataset" % dataset_name)
            for featurizer in self.featurizers:
                logging.info("Currently using featurizer: %s" % featurizer.name())
                dataset = self._load_dataset(dataset_name, featurizer)
                for splitter, target, training_size in self._split_dataset(dataset):
                    results = self._run_experiment(splitter, target, dataset)
                    experiment_results[dataset_name][featurizer.name()][training_size] = results
        self._dump_results(experiment_results)

    def _run_experiment(self, splitter, target, dataset):
        """Responsible for running all experiments specified in config."""
        experiment_results = defaultdict(list)
        for train, test in splitter:
            for experiment in self.experiments:
                # You might find yourself wondering why we're using lists here instead of np arrays
                # The answer is that pandas sucks.
                logging.info("Training: %s" % experiment.name())
                training_data = list(dataset['Features'].iloc[train])
                training_labels = list(dataset[target].iloc[train])
                experiment.train(training_data, training_labels)

                test_set = list(dataset['Features'].iloc[test])
                result = experiment.predict(test_set)
                ground_truth = list(dataset[target].iloc[test])
                logging.info("Measuring: %s" % experiment.name())

                metric_dict = self._measure_experiment(ground_truth, result)
                experiment_results[experiment.name()].append(metric_dict)
        return experiment_results

    def _measure_experiment(self, target, result):
        """Responsible for recording all metrics specified in config for a given experiment."""
        metric_results = defaultdict(dict)
        for metric in self.metrics:
            metric_results[metric.name()] = metric.evaluate(target, result)
        return metric_results

    @staticmethod
    def _dump_results(experiment_results):
        """Responsible for recording config and dumping experiment results in result directory."""
        current_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        result_path = "results/%s" % current_time
        if os.path.exists(result_path):
            raise ValueError("Result File %s already exists" % current_time)
        os.makedirs(result_path)

        result_file = "%s/Results.json" % result_path
        json.dump(experiment_results, open(result_file, 'w'))

        # The a is for archival, not just a typo
        config_record = "%s/Config.pya" % result_path
        copyfile("config.py", config_record)

    @staticmethod
    def _split_dataset(dataset):
        target_list = [column for column in dataset.columns.values if column.startswith("Target")]
        for training_size in TEST_SETUP["train_sizes"]:
            logging.info("Training with train set of size: %s" % training_size)
            # Sklearn technically offers a train_size parameter that seems like it would be better
            # Unfortunately it doesn't work as expected and locks test size to train size
            test_size = len(dataset) - training_size
            splitter = StratifiedShuffleSplit(TEST_SETUP["n_splits"], test_size=test_size)
            for target in target_list:
                # We can use np.zeros because it's stratifying the split
                # based on the categories. Removing the indexing saves time.
                yield splitter.split(np.zeros(len(dataset)), dataset[target]), target, training_size

    @staticmethod
    def _load_dataset(dataset_name, featurizer):
        """Responsible for loading a given dataset given the dataset_name and featurizer."""
        read_location = feature_set_location(dataset_name, featurizer)
        logging.info("Loading Dataset: %s" % read_location)
        # The literal_eval is because pandas doesn't believe in reading what it writes
        return pd.read_csv(read_location, converters={'Features': literal_eval})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Experimentation Started...")
    experimentation = Experimentation()
    logging.info("Running Experiments...")
    experimentation.run_experiments()
