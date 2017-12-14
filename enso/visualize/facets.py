"""Module for creating facet visualization."""
import os.path

import seaborn as sns
import matplotlib.patches as patches
from matplotlib import pyplot as plt

from enso.visualize import ClassificationVisualizer
from enso.config import RESULTS_DIRECTORY


class FacetGridVisualizer(ClassificationVisualizer):
    """Class for creating a grid of line graphs."""

    def visualize(
        self,
        results,
        x_tile,
        y_tile,
        x_axis,
        y_axis,
        lines,
        category=None,
        cv=None,
        results_id=None,
        **kwargs
    ):
        """Create a tiled visualization of experiment results."""
        sns.set(style="ticks", color_codes=True)
        y_limits = (min(results.Result.values), 1.0)
        grid = sns.FacetGrid(
            results,
            col=x_tile,
            row=y_tile,
            hue=lines,
            size=5,
            legend_out=False,
            margin_titles=True,
            sharey=True,
            ylim=y_limits,
            palette='deep'
        )
        grid = grid.map(sns.pointplot, x_axis, y_axis)
        colors = sns.color_palette('deep').as_hex()[:len(grid.hue_names)]
        handles = [patches.Patch(color=col, label=label) for col, label in zip(colors, grid.hue_names)]
        plt.legend(handles=handles)
        filename = os.path.join(RESULTS_DIRECTORY, results_id, "{}.png".format(self.__class__.__name__))
        plt.savefig(filename)
