"""Main method for running experiments as configured in config.py."""
import os
import logging
from shutil import copyfile
from time import gmtime, strftime
from ast import literal_eval
import abc
from functools import wraps
import multiprocessing
import itertools
import json

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
from tqdm import tqdm

from enso.resample import resample
from enso.sample import sample
from enso.utils import feature_set_location, get_plugins, BaseObject
from enso.config import FEATURIZERS, DATA, EXPERIMENTS, METRICS, TEST_SETUP, RESULTS_DIRECTORY, N_GPUS, N_CORES


POOL = ProcessPoolExecutor(N_CORES)


class Experimentation(object):
    """Responsible for running experiments configured in config."""

    def __init__(self, name=None):
        """Responsible for gathering and instantiating experiments, featurizers, and metrics."""
        self.name = name
        self.experiments = get_plugins("experiment", EXPERIMENTS, return_class=True)
        self.featurizers = get_plugins("featurize", FEATURIZERS)
        self.metrics = get_plugins("metrics", METRICS)
        self.columns = ['Dataset', 'Featurizer', 'Experiment', 'Metric', 'TrainSize', 'Sampler', 'Resampling', 'Result', 'TrainResult', 'Hyperparams']
        self.results = pd.DataFrame(columns=self.columns)

    def run_experiments(self):
        """Responsible for actually running experiments."""
        futures = {}
        for dataset_name in DATA:
            logging.info("Experimenting on %s dataset" % dataset_name)
            for featurizer in self.featurizers:
                logging.info("Currently using featurizer: %s" % featurizer.name())
                for training_size in TEST_SETUP["train_sizes"]:
                    for sampler in TEST_SETUP["samplers"]:
                        for resampling in TEST_SETUP["resamplings"]:
                            current_setting = {
                                'Dataset': dataset_name,
                                'Featurizer': featurizer.name(),
                                'TrainSize': training_size,
                                'Sampler': sampler,
                                'Resampling': resampling
                            }
                            future = POOL.submit(self._run_experiment, dataset_name, current_setting)
                            futures[future] = current_setting

        for future in concurrent.futures.as_completed(futures):
            current_setting = futures[future]
            try:
                results = future.result()
                logging.info("Finished training for {}".format(current_setting))
            except Exception:
                logging.exception("Exception occurred for {}".format(current_setting))

    def _run_experiment(self, dataset_name, current_setting):
        """Responsible for running all experiments specified in config."""
        results = pd.DataFrame(columns=self.columns)
        dataset = self._load_dataset(dataset_name, current_setting.get("Featurizer"))
        for splitter, target in self._split_dataset(dataset, current_setting['TrainSize']):
            for train_indices, test in splitter:
                train = sample(
                    current_setting['Sampler'],
                    dataset['Features'],
                    list(dataset[target].iloc[train_indices]),
                    train_indices,
                    current_setting['TrainSize']
                )
                for experiment_cls in self.experiments:
                    experiment = experiment_cls()

                    name = experiment.name()
                    internal_setting = {
                        'Experiment': name
                    }
                    internal_setting.update(current_setting)
                    print("Training with settings {}".format(internal_setting))
                    try:
                        # You might find yourself wondering why we're using lists here instead of np arrays
                        # The answer is that pandas sucks.
                        train_set = list(dataset['Features'].iloc[train])
                        train_labels = list(dataset[target].iloc[train])
                        test_set = list(dataset['Features'].iloc[test])
                        test_labels = list(dataset[target].iloc[test])
                        experiment.train(*resample(current_setting['Resampling'], train_set, train_labels))

                        test_pred = experiment.predict(test_set, subset='TEST')
                        train_pred = experiment.predict(train_set, subset='TRAIN')
                        result = self._measure_experiment(
                            target=test_labels,
                            result=test_pred,
                            train_target=train_labels,
                            train_result=train_pred,
                            internal_setting=internal_setting
                        )
                        self._dump_results(result, experiment_name=self.name)
                    except Exception:
                        logging.exception("Failed to run experiment: {}".format(internal_setting))
        return results

    def _measure_experiment(self, target, result, train_target=None, train_result=None, internal_setting=None, test_key='Result', train_key='TrainResult'):
        """Responsible for recording all metrics specified in config for a given experiment."""
        results = pd.DataFrame(columns=self.columns)
        for metric in self.metrics:
            score = metric.evaluate(target, result)
            full_setting = {"Metric": metric.name(), test_key: score}

            # measure score on train set to help detect overfitting
            if train_target is not None and train_result is not None:
                train_score = metric.evaluate(train_target, train_result)
                full_setting[train_key] = train_score

            full_setting.update(internal_setting)
            full_setting_df = pd.DataFrame.from_records([full_setting])
            results = results.append(full_setting_df, ignore_index=True)
        return results

    def _dump_results(self, results, experiment_name):
        """Responsible for recording config and dumping experiment results in result directory."""
        if not experiment_name:
            experiment_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        result_path = "%s/%s" % (RESULTS_DIRECTORY, experiment_name)
        results = results.dropna()

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        result_file = "%s/Results.csv" % result_path
        header = False if os.path.exists(result_file) else True
        result_fd = open(result_file, 'a')
        results.to_csv(result_fd, header=header, columns=self.columns)

        # The a is for archival, not just a typo
        config_record = "%s/Config.pya" % result_path
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config.py'
        ))
        copyfile(config_path, config_record)

    @staticmethod
    def _split_dataset(dataset, training_size):
        target_list = [column for column in dataset.columns.values if column.startswith("Target")]
        logging.info("Training with train set of size: %s" % training_size)
        # Sklearn technically offers a train_size parameter that seems like it would be better
        # Unfortunately it doesn't work as expected and locks test size to train size
        test_size = int(len(dataset) * TEST_SETUP["sampling_size"])
        if test_size + training_size > len(dataset):
            raise ValueError("Invalid training size provided.  Training size must be less than {} of dataset size.".format(TEST_SETUP["sampling_size"]))

        splitter = StratifiedShuffleSplit(TEST_SETUP["n_splits"], test_size=test_size)
        for target in target_list:
            # We can use np.zeros because it's stratifying the split
            # based on the categories. Removing the indexing saves time.
            yield splitter.split(np.zeros(len(dataset)), dataset[target]), target

    @staticmethod
    def _load_dataset(dataset_name, featurizer_name):
        """Responsible for loading a given dataset given the dataset_name and featurizer."""
        read_location = feature_set_location(dataset_name, featurizer_name)
        logging.info("Loading Dataset: %s" % read_location)
        return joblib.load(read_location)


class CheckOutput(object):
    """Decorator class to add output checks to different experiments classes."""

    def __new__(cls, *args):
        """Wrap predict method with appropriate decorator check."""
        experiment_class = super(CheckOutput, cls).__new__(cls, *args)
        experiment_class.predict = experiment_class.verify_output(experiment_class.predict)
        return experiment_class


class Experiment(BaseObject):
    """Base class for all Experiments."""

    __metaclass__ = CheckOutput

    param_grid = {}

    @abc.abstractmethod
    def train(self, dataset, target):
        """
        General endpoint to run training for a given experiment.

        dataset is a subselected version of the dataset to avoid test/train contamination.
        Target refers to the column of interest.

        Even if this method doesn't do anything it must be implemented
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, dataset):
        """
        General endpoint to predict on the test set for a given experiment.

        Dataset only contains relvant rows. Prediction format is dependant on class of experiment
        """
        raise NotImplementedError

    @classmethod
    def verify_output(cls, function):
        """Check predict output to ensure it complies with the experiment type."""
        return function


class ClassificationExperiment(Experiment):
    """Base class for classification experiments."""

    @classmethod
    def verify_output(cls, func):
        """Verify the output of classification tasks."""
        @wraps(func)
        def wrapped_predict(self, dataset):
            response = func(dataset)
            # Must have a prediction for each entry
            assert len(response) == len(dataset)
            # All probabilities should sum to approx. 1
            assert np.all(np.isclose(response.sum(axis=1), np.ones(len(response))))
            return response
        return func


class RegressionExperiment(Experiment):
    """Base class for regression experiments."""

    pass


class MatchingExperiment(Experiment):
    """Base class for matching experiments."""

    pass
