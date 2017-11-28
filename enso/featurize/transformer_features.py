import sys
import tensorflow as tf
import os
import os.path
import transformer

# TODO: change this to a dependency on the "transformer"
# package when implementation / saved weights are stable
import_path = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(transformer.__file__)),
        "data/models/SNLI"
    )
)
sys.path.append(import_path)

from v1.model import Transformer
from v1.config import Config
from v1.scripts.download_wmt import PAD_ID, GO_ID, UNK_ID
from v1.embedding import Glove
from indicoio.custom import vectorize
import numpy as np

from enso.featurize import Featurizer


def tokenize(tokens):
    Glove.load()
    return [Glove.token2idx.get(token, UNK_ID) for token in tokens]


class TransformerFeaturizer(Featurizer):

    initialized = False

    def __init__(self, *args, **kwargs):
        super(Featurizer).__init__(*args, **kwargs)
        TransformerFeaturizer.setup()

    @classmethod
    def setup(cls):
        if not cls.initialized:
            cls.initialized = True
            cls.config = Config()
            cls.config.max_length = 128 # pass in as a parameter?
            cls.transformer = Transformer()
            cls.transformer._build_snli(config=cls.config)
            saver = tf.train.Saver()

            cls.session = tf.Session()
            saver.restore(cls.session, '/transformer/data/models/SNLI/v0/2480000.ckpt')

    def featurize_list(self, series):
        example_batch = []
        lengths_batch = []
        glove_features = []
        for text in series.values:
            token_ids = tokenize(text.lower().split())[:TransformerFeaturizer.config.max_length]

            length = len(token_ids)
            delta = self.config.max_length - length
            token_ids += [PAD_ID] * delta

            example_batch.append(token_ids)
            lengths_batch.append(length)

        batch_size = 50
        for batch_start in range(0, len(series.values), batch_size):
            batch_end = batch_start + batch_size
            batch = series.values[batch_start:batch_end]
            glove_features.extend(vectorize(batch.tolist(), domain='standard'))

        glove_features = np.vstack(glove_features)

        transformer = TransformerFeaturizer.transformer
        session = TransformerFeaturizer.session

        embeddings = session.run(
            transformer.doc_a_embedding,
            feed_dict={
                transformer.a_token_idxs: example_batch,
                transformer.a_lengths: lengths_batch,
                transformer.dropout_rate: 0.0
            }
        )
        embeddings = np.hstack([glove_features, embeddings])
        return embeddings.tolist()
