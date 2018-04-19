"""Return ELMO features."""
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

from enso.featurize import Featurizer



class ElmoFeaturizer(Featurizer):

    def load(self):
        self.elmo_embedder = ElmoEmbedder()

    def featurize_list(self, dataset, batch_size=10):
        all_features = []
        for batch_start in tqdm(range(0, len(dataset), batch_size)):
            batch_docs = list(dataset[batch_start:batch_start + batch_size])
            batch_tokens = [word_tokenize(doc) for doc in batch_docs]
            embeddings = self.elmo_model.embed_batch(batch_tokens)
            embeddings = [
                np.hstack(
                    [embedding[i,:,:] for i in range(embedding.shape[0])]
                ) for embedding in embeddings
            ]
            all_features.extend(embeddings)
        return all_features
