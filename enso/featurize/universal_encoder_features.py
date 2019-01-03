import tensorflow as tf
import tensorflow_hub as hub

from enso import Featurizer
from enso.mode import ModeKeys
from enso.registry import Registry


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