"""Module for creating facet visualization."""
import os.path
import json

import seaborn as sns
import matplotlib.patches as patches
from matplotlib import pyplot as plt

from enso.visualize import ClassificationVisualizer
from enso.config import RESULTS_DIRECTORY


class FacetGridVisualizer(ClassificationVisualizer):
    """
    Create a grid of line graphs based on the value of `config.VISUALIZATION_OPTIONS`
    """

    def visualize(
        self,
        results,
        x_tile,
        y_tile,
        x_axis,
        y_axis,
        lines,
        results_id=None,
        filename='FacetGridVisualizer',
        **kwargs
    ):
        """
        Create a tiled visualization of experiment results.

        :param results: pd.DataFrame of results, loaded from results .csv file.
        :param x_tile: string name of DataFrame column to vary over the x axis of the grid of line graphs
        :param y_tile: string name of DataFrame column to vary over the y axis of the grid of line graphs
        :param x_axis: string name of DataFrame column to plot on the x axis within each individual line graph
        :param y_axis: string name of DataFrame column to plot on the y axis within each individual line graph
        :param lines: string name or list of DataFrame column string names displayed as separate lines within each graph.
                      Providing multiple values means that each unique combination of values will be displayed as a single line.
        :param results_id: string name of folder to save resulting visual in, relative to the root of the results directory
        :param filename: filename (excluding filetype) to use when saving visualization.  Value is relative to folder specified by results_id.
        """
        sns.set(style="ticks", color_codes=True)

        if isinstance(lines, (tuple, list)):
            results['key'] = results[lines].apply(lambda x: ','.join(x), axis=1)
            lines = 'key'

        n_lines = len(results[lines].unique())
        rc = {'lines.linewidth': 1, 'lines.markersize': 1}
        sns.set_context("paper", rc=rc)
        grid = sns.FacetGrid(
            results,
            col=x_tile,
            row=y_tile,
            hue=lines,
            size=5,
            legend_out=False,
            margin_titles=True,
            sharey=False,
            sharex=False,
            palette=sns.color_palette("hls", n_lines)
        )
        grid = grid.map(sns.pointplot, x_axis, y_axis)
        colors = sns.color_palette("hls", n_lines).as_hex()[:len(grid.hue_names)]
        handles = [patches.Patch(color=col, label=label) for col, label in zip(colors, grid.hue_names)]
        plt.legend(handles=handles)
        plt.tight_layout()
        filename = os.path.join(RESULTS_DIRECTORY, results_id, "{}.png".format(filename))
        plt.savefig(filename)
