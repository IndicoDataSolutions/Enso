import numpy as np

import spacy

from bayesianesque import EmbedModel
#from gensim.models import KeyedVectors

from spacy.cli.download import download

from enso.featurize import Featurizer
from enso.registry import Registry, ModeKeys

#import tensorflow_hub as hub
import tensorflow as tf


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

@Registry.register_featurizer(ModeKeys.ANY)
class BayesianesqueFeaturizer(Featurizer):
    def load(self):
        """
        If the pre-trained `en_vectors_web_lg` model is not already stored on disk, it will be automatically downloaded
        as part of :func:`load()`. Note that the download process may require sudo permissions depending on your python package settings.
        """
        self.nlp = EmbedModel()

    def featurize(self, x):
        return self.nlp.predict([x])[0]

    def featurize_batch(self, X, **kwargs):
        return self.nlp.predict(X)

@Registry.register_featurizer(ModeKeys.ANY)
class Word2Vec(Featurizer):
    def load(self):
        self.model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def featurize(self, text):
        tokens = text.split(" ")
        i = 0
        vecs = [np.zeros(300)]
        while i < len(tokens):
            for j in range(len(tokens), i, -1):
                n_gram = "_".join(tokens[i:j])
                if n_gram in self.model:
                    vecs.append(self.model[n_gram])
                    i = j
                    break
            i += 1
        return np.mean(vecs, axis=0)

    def featurize_batch(self, X, **kwargs):
        return [self.featurize(x) for x in X]

@Registry.register_featurizer(ModeKeys.ANY)
class UniversalEncoder(Featurizer):
    def load(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        session = tf.Session(config=config)
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        input_placeholder = tf.placeholder(tf.string, shape=(None))
        embedded = embed(input_placeholder)
        self.model = lambda list_of_text: session.run(embedded, feed_dict={input_placeholder: list_of_text})

    def featurize(self, text):
        return self.model([text])

    def featurize_batch(self, X, **kwargs):
        all_features = []
        batch_size = 100
        for i in range(0, len(X), batch_size):
            chunk_data = list(X[i:i + batch_size])
            all_features.extend(self.model(chunk_data))
        return all_features

@Registry.register_featurizer(ModeKeys.ANY)
class UniversalEncoderLg(Featurizer):
    def load(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        session = tf.Session(config=config)
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        input_placeholder = tf.placeholder(tf.string, shape=(None))
        embedded = embed(input_placeholder)
        self.model = lambda list_of_text: session.run(embedded, feed_dict={input_placeholder: list_of_text})

    def featurize(self, text):
        return self.model([text])

    def featurize_batch(self, X, **kwargs):
        all_features = []
        batch_size = 100
        for i in range(0, len(X), batch_size):
            chunk_data = list(X[i:i + batch_size])
            all_features.extend(self.model(chunk_data))
        return all_features
