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
            non_timed_results = results[~results.Metric.isin(['train_time', 'pred_time'])]
            cols = lines + [x_tile] + [y_tile]
            grouped_results = non_timed_results.groupby(cols).Result.max()
            subsetted_results = []
            for setting, value in zip(grouped_results.index.to_list(), grouped_results.values):
                best_hparams = results[(results.Result == value) &
                                       (results[cols] == setting).all(1)][pick_best].values[0]
                print(setting, best_hparams)
                full_setting = list(setting) + list(best_hparams)
                full_cols = cols + pick_best
                subsetted_results.append(results[(results[full_cols] == full_setting).all(1)])
        # append time results back in
        subsetted_results.append(results[results.Metric.isin(['train_time', 'pred_time'])])
        new_results = pd.concat(subsetted_results, axis=0)
        print(new_results[new_results.Dataset == "SequenceLabeling/invoices"][new_results.TrainSize == 100])
        print(new_results[new_results.Metric == "MacroCharF1"][new_results.TrainSize == 100][['Dataset', 'Experiment', 'Result']])
        super().visualize(
                new_results,
                x_tile,
                y_tile,
                x_axis,
                y_axis,
                lines,
                results_id=results_id,
                filename=filename)
