"""Module for creating facet visualization."""
import os.path
import json

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
        filename='FacetGridVisualizer',
        **kwargs
    ):
        """Create a tiled visualization of experiment results."""
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
