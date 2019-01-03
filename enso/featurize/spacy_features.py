import spacy
from spacy.cli.download import download

from enso.featurize import Featurizer
from enso.registry import Registry, ModeKeys


@Registry.register_featurizer(ModeKeys.ANY)
class SpacyCNNFeaturizer(Featurizer):
    """
    Featurizer that embeds documents using the mean of spacy's text CNN embedding model.

    """

    def load(self):
        """
        If the pre-trained `en_core_web_lg` model is not already stored on disk, it will be automatically downloaded
        as part of :func:`load()`. Note that the download process may require sudo permissions depending on your python package settings.
        """
        try:
            self.nlp = spacy.load('en_core_web_lg')
        except OSError:
            download('en_core_web_lg')
            self.nlp = spacy.load('en_core_web_lg')

    def featurize(self, x):
        return self.nlp(x).vector

    def featurize_batch(self, X, **kwargs):
        return [self.nlp(x).vector for x in X]


@Registry.register_featurizer(ModeKeys.ANY)
class SpacyGloveFeaturizer(Featurizer):
    """
    Featurizer that embeds documents using the mean of a document's glove vectors.
    """

    def load(self):
        """
        If the pre-trained `en_vectors_web_lg` model is not already stored on disk, it will be automatically downloaded
        as part of :func:`load()`. Note that the download process may require sudo permissions depending on your python package settings.
        """
        try:
            self.nlp = spacy.load('en_vectors_web_lg')
        except OSError:
            download('en_vectors_web_lg')
            self.nlp = spacy.load('en_vectors_web_lg')

    def featurize(self, x):
        return self.nlp(x).vector

    def featurize_batch(self, X, **kwargs):
        return [self.nlp(x).vector for x in X]


