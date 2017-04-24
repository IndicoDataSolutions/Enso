"""Main file for creating visualizations."""
import logging
import pandas as pd

from config import VISUALIZATIONS, VISUALIZATION_OPTIONS
from utils import get_plugins, get_all_experiment_runs


class Visualization(object):
    """Object for visualization orchestration."""

    def __init__(self, test_run=None):
        """Visualize results from a given test run."""
        self.results = self._grab_results(test_run)
        self.visualizations = get_plugins("visualize", VISUALIZATIONS)

    @staticmethod
    def _grab_results(test_run):
        """
        Grab the appropriate results file.

        `test_run` can be set either to None, which will default to the most recent run,
        a string, which will search for a match timestamp in the results directory, or
        an integer `n` which will grab the test run from `n` runs ago. 0 would be the same as None.
        """
        all_runs = get_all_experiment_runs()
        correct_run = None
        if test_run is None:
            correct_run = all_runs[0]
        elif isinstance(test_run, int):
            correct_run = all_runs[test_run]
        elif isinstance(test_run, str):
            if test_run in all_runs:
                correct_run = test_run
            else:
                raise ValueError("Experiment run: %s not found" % test_run)
        else:
            raise ValueError("test_run must be either None, an int, or a string")
        return pd.read_csv('results/%s/Results.csv' % correct_run)

    def visualize(self):
        """Pass visualization options defined in config to instantiated visualizations."""
        global_options = {a: b for a, b in VISUALIZATION_OPTIONS.items() if not isinstance(b, dict)}
        global_options['results'] = self.results
        for visualization in self.visualizations:
            local_options = VISUALIZATION_OPTIONS.get(visualization.name(), {})
            local_options.update(global_options)
            visualization.visualize(**local_options)


if __name__ == "__main__":
    logging.info('Loading Results...')
    visualization = Visualization()
    logging.info('Painting Pretty Pictures...')
    visualization.visualize()
