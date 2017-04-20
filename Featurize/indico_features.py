"""File for storing the featurizers that indico offers via API."""
from featurize import Featurizer
from indicoio.custom import vectorize


CHUNK_SIZE = 100


def featurization_factory(domain):
    """Responsible for creating indico featurization functions."""
    @staticmethod
    def indico_feature_func(dataset):
        all_features = []
        for i in xrange(0, len(dataset), CHUNK_SIZE):
            chunk_data = list(dataset[i:i + CHUNK_SIZE])
            all_features.extend(vectorize(chunk_data, domain=domain))
        return all_features
    return indico_feature_func


class IndicoStandard(Featurizer):
    """Featurizer that uses indico's standard features."""

    featurize_list = featurization_factory("standard")


class IndicoSentiment(Featurizer):
    """Featurizer that uses indico's sentiment features."""

    featurize_list = featurization_factory("sentiment")


class IndicoTopics(Featurizer):
    """Featurizer that uses indico's topics features."""

    featurize_list = featurization_factory("topics")


class IndicoFinance(Featurizer):
    """Featurizer that uses indico's finance features."""

    featurize_list = featurization_factory("finance")
