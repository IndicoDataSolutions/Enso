"""Module for creating facet visualization."""
import os.path
import json

import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import pandas as pd

from enso.visualize.facets import FacetGridVisualizer
from enso.config import RESULTS_DIRECTORY
from enso.registry import Registry


@Registry.register_visualizer()
class FacetGridBestVisualizer(FacetGridVisualizer):
    def visualize(
            self,
            results,
            x_tile,
            y_tile,
            x_axis,
            y_axis,
            lines,
            pick_best,
            metric,
            results_id=None,
            filename='FacetGridBestVisualizer',
            **kwargs
    ):
        if pick_best:
            cols = lines + [x_tile] + [y_tile]
            grouped_results = results.groupby(cols).Result.max()
            subsetted_results = []
            for setting in grouped_results.index.to_list():
                subsetted_results.append(results[(results[cols] == setting).all(1)])
        new_results = pd.concat(subsetted_results, axis=0)
        super().visualize(
                new_results,
                x_tile,
                y_tile,
                x_axis,
                y_axis,
                lines,
                results_id=results_id,
                filename=filename)