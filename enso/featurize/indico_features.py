"""File for storing the featurizers that indico offers via API."""
from indicoio.custom import vectorize
import indicoio.config
from tqdm import tqdm

from enso.featurize import Featurizer

# indicoio.config.url_protocol = "http"
# indicoio.config.host = "localhost:8000"

CHUNK_SIZE = 50


def featurization_factory(domain, **kwargs):
    """Responsible for creating indico featurization functions."""
    @staticmethod
    def indico_feature_func(dataset):
        all_features = []
        for i in tqdm(range(0, len(dataset), CHUNK_SIZE)):
            chunk_data = list(dataset[i:i + CHUNK_SIZE])
            all_features.extend(vectorize(chunk_data, domain=domain, **kwargs))
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


class IndicoTransformer(Featurizer):
    """Featurizer that uses indico's mean pooled transformer features."""

    featurize_list = featurization_factory("transformer")


class IndicoTransformerSequence(Featurizer):
    """Featurizer that uses indico's transformer sequence features."""

    featurize_list = featurization_factory("transformer", sequence=True)


class IndicoStandardSequence(Featurizer):
    """Featurizer that uses indico's standard sequence features"""

    featurize_list = featurization_factory("standard", sequence=True)


class IndicoEmotion(Featurizer):
    """Featurizer that uses indico's finance features."""

    featurize_list = featurization_factory("emotion")


class IndicoFastText(Featurizer):
    """Featurizer that uses indico's finance features."""

    featurize_list = featurization_factory("fasttext")

class IndicoElmo(Featurizer):
    """Featurizer that uses indico's finance features."""

    featurize_list = featurization_factory("elmo")