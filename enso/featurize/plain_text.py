"""File for storing the featurizers that indico offers via API."""
from indicoio.custom import vectorize

from enso.featurize import Featurizer


class PlainTextFeaturizer(Featurizer):
    """
    Essentially a no op -- return raw text.  
    Featurizer placeholder to support training from scratch or model finetuning comparison.
    """

    def featurize_batch(self, X, **kwargs):
        """
        :param X: `pd.Series` that contains raw text to featurize
        :param batch_size: int number of examples to process per batch
        :returns: list of np.ndarray representations of text
        """
        return X