"""Base class for visualization objects."""
from utils import BaseObject, get_all_experiment_runs


class Visualization(BaseObject):
    """Base class for creating visualizations."""

    def create_plot(self, results, display=True, write=True):
        """
        Create visualization for the given test_run.

        `display`=True will default to showing the generated visualizations as they are created.
        `write`=True will default to saving the generated image in the Results directory.
        """
        pass

    @staticmethod
    def _get_results_dir(test_run):
        """Grab the appropriate results directory."""
        get_all_experiment_runs()
