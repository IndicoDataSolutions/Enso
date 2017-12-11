"""Main file for creating visualizations."""
import logging
import abc
import ast
import json
from functools import wraps
import os.path

import pandas as pd
import numpy as np
import matplotlib

from enso.config import VISUALIZATIONS, VISUALIZATION_OPTIONS, RESULTS_DIRECTORY
from enso.utils import get_plugins, get_all_experiment_runs
from enso.utils import BaseObject

# set matplotlib up for headless environment
matplotlib.use("Agg")


class Visualization(object):
    """Object for visualization orchestration."""

    def __init__(self, test_run=None):
        """Visualize results from a given test run."""
        self.results_id = self._resolve_results_id(test_run)
        self.results = self._grab_results(self.results_id)
        self.visualizers = get_plugins("visualize", VISUALIZATIONS)

    @staticmethod
    def _resolve_results_id(test_run):
        """
        Grab the appropriate results file id.
        `test_run` can be set either to None, which will default to the most recent run,
        a string, which will search for a match timestamp in the results directory, or
        an integer `n` which will grab the test run from `n` runs ago. 0 would be the same as None.
        """
        correct_run = None
        if isinstance(test_run, str):
            correct_run = test_run
            if not os.path.exists(os.path.join(RESULTS_DIRECTORY, correct_run)):
                raise ValueError("Experiment run: %s not found" % test_run)
        elif test_run is None:
            all_runs = get_all_experiment_runs()
            correct_run = all_runs[0]
        elif isinstance(test_run, int):
            correct_run = all_runs[test_run]
        else:
            raise ValueError("test_run must be either None, an int, or a string")
        return correct_run

    @staticmethod
    def _grab_results(results_id):
        """
        Read data file that corresponds with the provided results id and return
        resulting pd.DataFrame() instance.
        """
        return pd.read_csv('{}/{}/Results.csv'.format(RESULTS_DIRECTORY, results_id))

    def visualize(self):
        """Pass visualization options defined in config to instantiated visualizations."""
        # Grabs all settings that don't have a dict nested below.
        global_options = {a: b for a, b in VISUALIZATION_OPTIONS.items() if not isinstance(b, dict)}
        global_options['results'] = self.results
        global_options['results_id'] = self.results_id
        for visualizer in self.visualizers:
            local_options = VISUALIZATION_OPTIONS.get(visualizer.name(), {})
            local_options.update(global_options)
            visualizer.visualize(**local_options)


class Visualizer(BaseObject):
    """Base class for creating visualizations."""

    @abc.abstractmethod
    def visualize(self, results, display=True, write=True):
        """
        Create visualization for the given test_run.
        `display`=True will default to showing the generated visualizations as they are created.
        `write`=True will default to saving the generated image in the Results directory.
        """
        raise NotImplementedError


class DataHandler(abc.ABCMeta):
    """Handles cross-validation and category strategies on results before passing them on."""

    def __new__(cls, *args, **kwargs):
        """Decorate visualize method with cv and category handlers."""
        viz_class = super(DataHandler, cls).__new__(cls, *args, **kwargs)
        viz_class.visualize = viz_class.handle_categories(viz_class.visualize)
        viz_class.visualize = viz_class.handle_cv(viz_class.visualize)
        return viz_class


class ClassificationVisualizer(Visualizer):
    """Base class for classification visualizations."""

    __metaclass__ = DataHandler

    @classmethod
    def handle_categories(cls, func):
        """
        Execute a category strategy on a result set.
        Force the user to make a choice about handling predictions for different classes.
        It should support either being one of the axes for the Main Visualization,
        or there should be a strategy for turning multiple entries into a single one
        """
        @wraps(func)
        def wrapped_visualize(self, results, **kwargs):
            category = kwargs.get('category', None)
            if category is not None:
                class_info = [ast.literal_eval(result) for result in results['Result']]
                new_info = []
                result = results.copy()
                if category == 'merge':
                    for entry in class_info:
                        sample_value = entry.values()[0]
                        # Check if all entries are basically identical
                        if np.allclose(entry.values(), sample_value):
                            new_info.append(sample_value)
                        else:
                            raise ValueError("`merge` is not a valid strategy if classes vary.")
                    result['Result'] = new_info
                else:
                    raise ValueError("`%s` strategy not found" % category)
                return func(results, **kwargs)
            elif 'Category' not in kwargs.values():
                raise ValueError("""
                    Category must either have a strategy listed, or be a displayed variable.
                """)
        return func

    @classmethod
    def handle_cv(cls, func):
        """
        Execute a cv strategy on a result_set.
        Same as handle_categories, but for multiple cv runs rather than multiple classes
        """
        @wraps(func)
        def wrapped_visualize(self, results, **kwargs):
            cv = kwargs.get('cv', None)
            if cv is not None:
                new_results = pd.DataFrame(columns=results.columns.values)
                for row_set in self._iterate_identical_rows(results):
                    sample_row = row_set.iloc[0]
                    if cv == "mean":
                        if row_set['Result'].dtype == np.float64:
                            mean = row_set['Result'].mean()
                            sample_row['Result'] = mean
                        elif row_set['Result'].dtype == np.object:
                            values = [ast.literal_eval(item) for item in row_set['Result']]
                            average_dict = {}
                            # Grabbing from first entry since they should all be identical
                            for category in values[0]:
                                all_values = [value[category] for value in values]
                                average_dict[category] = sum(all_values) / len(all_values)
                            sample_row['Result'] = json.dumps(average_dict)
                        new_results = new_results.append(sample_row, ignore_index=True)
                    else:
                        raise ValueError("mean is the only cv thing we support right now")
                return func(new_results, **kwargs)
            elif 'Split' not in kwargs.values():
                raise ValueError("""
                    Split must either have a strategy listed, or be a displayed variable.
                """)
        return func

    @classmethod
    def _iterate_identical_rows(self, results):
        """Generate each set of rows identical other than their results column."""
        found_rows = set()
        for _, row in results.iterrows():
            current_row = self._row_repr(row)
            if current_row not in found_rows:
                found_rows.add(current_row)
                yield results.loc[
                    (results['Dataset'] == row['Dataset']) &
                    (results['Experiment'] == row['Experiment']) &
                    (results['Metric'] == row['Metric']) &
                    (results['TrainSize'] == row['TrainSize']) &
                    (results['Featurizer'] == row['Featurizer'])
                ]

    @staticmethod
    def _row_repr(row):
        """Return a tuple representation of a row without the Result colum."""
        return (
            row['Dataset'],
            row['Experiment'],
            row['Metric'],
            row['TrainSize'],
            row['Featurizer']
        )
