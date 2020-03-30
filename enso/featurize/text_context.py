import os
import joblib
import pandas as pd

from enso.featurize import Featurizer
from enso.utils import feature_set_location

class TextContextFeaturizer(Featurizer):
    """
    Takes a list of dictionaries, serialized as jsons, and converts them to
    a pandas DataFrame of text, features, and labels

    Input:
    [{
        'text': <str>,
        'context': <list of dicts>,
        'labels': <list of dicts>
    },
    ...]

    Output: columns are
    - Text: string text
    - Features: a tuple of the text and context
    - Labels: a list of dicts
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, dataset, dataset_name):
        """
        Given a `dataset` pandas DataFrame and string `dataset_name`,
        add column `"Features"` to the provided `pd.DataFrame` and serialize the result
        to the results folder listed in `config.py`.

        If a given featurizer exposes a :func:`featurize_batch` method, that method will be
        called to perform featurization.  Otherwise, :class:`Featurizer`'s will fall
        back to calling :func:`featurize` on each individual example.

        :param dataset: `pd.DataFrame` object that must contain a `Text` column.
        :param dataset_name: `str` name to use as a save location in the `config.FEATURES_DIRECTORY`.
        """

        if os.path.exists(feature_set_location(dataset_name, self.__class__.__name__)):
            print("Skipping, already have this feature combination.")
            return

        if type(dataset) != list:
            raise ValueError("dataset must be a list")
        text = [d['text'] for d in dataset]
        features = [(d['text'], d['context']) for d in dataset]
        new_dataset = pd.DataFrame(data={
            "Text": text,
            "Targets": [d['labels'] for d in dataset],
            "Features": features
        })
        self._write(new_dataset, dataset_name)