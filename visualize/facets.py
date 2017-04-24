"""Module for creating facet visualization."""
import ast
import numpy as np
import pandas as pd
import seaborn as sns

from visualize import Visualization


class FacetGridVisualization(Visualization):
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
        named_params = {x_tile, y_tile, x_axis, y_axis, lines}  # This is a set
        if category is not None:
            results = self.handle_categories(results, category)
        elif 'Category' not in named_params:
            raise ValueError("""
                Category must either have a strategy listed, or be a displayed variable.
            """)

        if cv is not None:
            results = self.handle_cv(results, cv)
        elif 'Split' not in named_params:
            raise ValueError("""
                Split must either have a strategy listed, or be a displayed variable.
            """)

        sns.set(style="ticks", color_codes=True)
        grid = sns.FacetGrid(results, col=x_tile, row=y_tile, hue=lines)
        grid = grid.map(sns.pointplot, x_axis, y_axis)
        sns.plt.show()

    @classmethod
    def handle_categories(self, results, category):
        """Resolve multiple categories into a single entry."""
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
        return result

    @classmethod
    def handle_cv(self, results, cv):
        """Resolve multiple CV runs into a single entry."""
        master_list = pd.DataFrame(columns=results.columns.values)
        for row_set in self._iterate_identical_rows(results):
            sample_row = row_set.iloc[0]
            if cv == "mean":  # TODO: support non-merged classes
                mean = row_set['Result'].mean()
                sample_row['Result'] = mean
                master_list = master_list.append(sample_row, ignore_index=True)
            else:
                raise ValueError("mean is the only cv thing we support right now")
        return master_list

    @classmethod
    def _iterate_identical_rows(self, results):
        """Generate each set of rows identical other than their results column."""
        found_rows = set()
        for _, row in results.iterrows():
            current_row = self._row_repr(row)
            if current_row not in found_rows:
                found_rows.update([current_row])
                yield results.loc[
                    (results['Dataset'] == row['Dataset']) &
                    (results['Experiment'] == row['Experiment']) &
                    (results['Metric'] == row['Metric']) &
                    (results['TrainSize'] == row['TrainSize']) &
                    (results['Featurizer'] == row['Featurizer'])
                ]

    @staticmethod
    def _row_repr(row):
        """Return a string representation of a row without the Result colum."""
        non_result_elements = (
            row['Dataset'],
            row['Experiment'],
            row['Metric'],
            row['TrainSize'],
            row['Featurizer']
        )
        return "%s%s%s%s%s" % (non_result_elements)
