"""Module for creating facet visualization."""
import os.path
import json

import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from matplotlib import pyplot as plt

from enso.visualize import ClassificationVisualizer
from enso.config import RESULTS_DIRECTORY
from enso.registry import Registry


@Registry.register_visualizer()
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
            for col in lines:
                results[col] = results[col].astype(str)
            results['key'] = results[lines].apply(lambda x: ','.join(x), axis=1)
            lines = 'key'
        y_tiles = np.unique(results[y_tile])
        x_tiles = np.unique(results[x_tile])
        keys = np.unique(results.key)
        colors_dict = {key: color for key, color in zip(keys, sns.color_palette("hls", len(keys)))}
        n_y_tiles = len(y_tiles)
        n_x_tiles = len(x_tiles)
        # we adjust the figsize based on how many plots will be plotted
        # we maintain a 6:8 ratio of height to width for uniformity
        fig, axes = plt.subplots(n_y_tiles, n_x_tiles, figsize=(n_x_tiles * 8, n_y_tiles * 6), squeeze=False)
        for i, row in enumerate(y_tiles):
            for j, col in enumerate(x_tiles):
                ax = axes[i][j]
                results_subset = results[(results[y_tile] == y_tiles[i]) & (results[x_tile] == x_tiles[j])]
                ax.set_xlabel(y_tiles[i])
                ax.set_ylabel(x_tiles[j])
                for key in keys:
                    line = results_subset[results_subset.key == key]
                    mean_results = line.groupby(x_axis)[y_axis].apply(np.mean)
                    sd_results = line.groupby(x_axis)[y_axis].apply(np.std)
                    ax.errorbar(mean_results.index, mean_results.values, yerr=sd_results, color=colors_dict[key],
                                label=key)
        # each Axes object will have the same handles and labels
        handles, labels = axes[0][0].get_legend_handles_labels()
        # the hard-coded numbers scale with the size of the plot
        legend = axes[-1][-1].legend(handles, labels, loc = 'best', bbox_transform=fig.transFigure)
        fig.tight_layout()
        filename = os.path.join(RESULTS_DIRECTORY, results_id, "{}.png".format(filename))
        plt.savefig(filename, bbox_extra_artists=(legend,), bbox_inches='tight')
