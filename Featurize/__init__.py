"""Module for all featurization methods that one might want to test."""
import pandas as pd


class Featurizer(object):
    """Base class for building featurizers."""

    def generate(self, dataset, dataset_name):
        """Responsible for generating appropriately-named feature datasets."""
        features = []
        if callable(getattr(self, "featurize_list", None)):
            features = self.featurize_list(dataset['Text'])
        elif callable(getattr(self, "featurize", None)):
            features = [self.featurize(entry) for entry in dataset['Text']]
        else:
            raise NotImplementedError("""
                Featurizers must implement the featurize_list, or the featurize method
            """)
        new_dataset = dataset.copy()  # Don't want to modify the underlying dataframe
        new_dataset['Text'] = features
        new_dataset.rename(columns={'Text': 'Features'}, inplace=True)
        self._write(new_dataset, dataset_name)

    def _write(self, featurized_dataset, dataset_name):
        """Responsible for taking a featurized dataset and writing it out to the filesystem."""
        base_dir, _, filename = dataset_name.rpartition('/')
        write_location = "Features/%s/" % base_dir
        class_name = self.__class__.__name__
        dump_name = "%s_%s_features.csv" % (filename, class_name)
        featurized_dataset.to_csv(write_location + dump_name)
