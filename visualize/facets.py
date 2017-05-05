"""Module for creating facet visualization."""
import seaborn as sns

from visualize import ClassificationVisualization


class FacetGridVisualization(ClassificationVisualization):
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
        **kwargs
    ):
        """Create a tiled visualization of experiment results."""
        sns.set(style="ticks", color_codes=True)
        grid = sns.FacetGrid(results, col=x_tile, row=y_tile, hue=lines)
        grid = grid.map(sns.pointplot, x_axis, y_axis)
        sns.plt.show()
