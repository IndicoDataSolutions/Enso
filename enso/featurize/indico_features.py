"""File for storing the featurizers that indico offers via API."""
from indicoio.custom import vectorize
import indicoio.config
from tqdm import tqdm

from enso.featurize import Featurizer


class BaseIndicoFeaturizer(Featurizer):
    """
    Base model from which all indico `Featurizer`'s inherit.
    """
    domain = None
    sequence = False

    def featurize_batch(self, X, batch_size=32, **kwargs):
        """
        :param X: `pd.Series` that contains raw text to featurize
        :param batch_size: int number of examples to process per batch
        :returns: list of np.ndarray representations of text
        """
        all_features = []
        for i in tqdm(range(0, len(X), batch_size)):
            chunk_data = list(X[i:i + batch_size])
            all_features.extend(vectorize(
                chunk_data, domain=self.domain, sequence=self.sequence, **kwargs
            ))
        return all_features


class IndicoStandard(BaseIndicoFeaturizer):
    """Featurizer that uses indico's standard features."""
    domain = 'standard'


class IndicoSentiment(BaseIndicoFeaturizer):
    """Featurizer that uses indico's sentiment features."""
    domain = 'sentiment'


class IndicoTopics(BaseIndicoFeaturizer):
    """Featurizer that uses indico's topics features."""
    domain = 'topics'


class IndicoFinance(BaseIndicoFeaturizer):
    """Featurizer that uses indico's finance features."""
    domain = 'finance'


class IndicoTransformer(BaseIndicoFeaturizer):
    """Featurizer that uses indico's transformer features."""
    domain = 'transformer'


class IndicoEmotion(BaseIndicoFeaturizer):
    """Featurizer that uses indico's emotion features."""
    domain = 'emotion'


class IndicoFastText(BaseIndicoFeaturizer):
    """Featurizer that uses indico's fasttext features."""
    domain = 'fasttext'


class IndicoElmo(BaseIndicoFeaturizer):
    """Featurizer that uses indico's finance features."""
    domain = 'elmo'


# NOTE: To remain undocumented for now until further API design decisions are made
class IndicoElmoSequence(BaseIndicoFeaturizer):
    """Featurizer that uses indico's finance features."""
    domain = 'elmo'
    sequence = True


class IndicoTransformerSequence(BaseIndicoFeaturizer):
    """Featurizer that uses indico's transformer sequence features."""
    domain = 'transformer'
    sequence = True


class IndicoStandardSequence(BaseIndicoFeaturizer):
    """Featurizer that uses indico's standard sequence features"""
    domain = 'standard'
    sequence = True
