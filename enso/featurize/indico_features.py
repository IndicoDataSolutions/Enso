"""File for storing the featurizers that indico offers via API."""
from indicoio.custom import vectorize
import indicoio.config
from tqdm import tqdm

from enso.featurize import Featurizer


def featurization_factory(domain, **kwargs):
    """Responsible for creating indico featurization functions."""
    @staticmethod
    def indico_feature_func(dataset, batch_size):
        all_features = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            chunk_data = list(dataset[i:i + batch_size])
            all_features.extend(vectorize(chunk_data, domain=domain, **kwargs))
        return all_features
    return indico_feature_func


class IndicoStandard(Featurizer):
    """Featurizer that uses indico's standard features."""

    featurize_batch = featurization_factory("standard")


class IndicoSentiment(Featurizer):
    """Featurizer that uses indico's sentiment features."""

    featurize_batch = featurization_factory("sentiment")


class IndicoTopics(Featurizer):
    """Featurizer that uses indico's topics features."""

    featurize_batch = featurization_factory("topics")


class IndicoFinance(Featurizer):
    """Featurizer that uses indico's finance features."""

    featurize_batch = featurization_factory("finance")


class IndicoTransformer(Featurizer):
    """Featurizer that uses indico's transformer features."""

    featurize_batch = featurization_factory("transformer")


class IndicoEmotion(Featurizer):
    """Featurizer that uses indico's emotion features."""

    featurize_batch = featurization_factory("emotion")


class IndicoFastText(Featurizer):
    """Featurizer that uses indico's fasttext features."""

    featurize_batch = featurization_factory("fasttext")



# To remain undocumented for now until further API design decisions are made
class IndicoElmo(Featurizer):
    """Featurizer that uses indico's finance features."""

    featurize_batch = featurization_factory("elmo")


class IndicoTransformerSequence(Featurizer):
    """Featurizer that uses indico's transformer sequence features."""

    featurize_batch = featurization_factory("transformer", sequence=True)


class IndicoStandardSequence(Featurizer):
    """Featurizer that uses indico's standard sequence features"""

    featurize_batch = featurization_factory("standard", sequence=True)

