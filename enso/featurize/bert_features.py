import os
import json
import logging

import pandas as pd
import random
import torch
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from indicoio.custom import Collection
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertModel, BertForMaskedLM, BertAdam
from indicoio.custom import vectorize
import indicoio.config

from enso.featurize import Featurizer
from enso.experiment import ClassificationExperiment
from enso.config import RESULTS_DIRECTORY
from enso.registry import Registry, ModeKeys


logger = logging.getLogger(__file__)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertModel.from_pretrained(
    'bert-base-uncased',
    cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
)
DEVICE = 'cuda'
model.to(DEVICE)
model.eval()


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


def convert_examples_to_features(examples, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in examples:
        tokens = tokenizer.tokenize(example)

        # Truncate tokens if too long
        if len(tokens) > (max_seq_length - 2):
            tokens = tokens[:(max_seq_length - 2)]
        
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
            )
        )
    return features

@Registry.register_featurizer(ModeKeys.CLASSIFY)
class BERTFeaturizer(Featurizer):
    """
    Base model from which all indico `Featurizer`'s inherit.
    """

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        
        ############
        # SETTINGS #
        ############
        self.batch_size = 4
        self.max_seq_length = 512
        ############
        
    def _to_dataloader(self, X):
        features = convert_examples_to_features(
            X, self.max_seq_length
        )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask)
        sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)

    def featurize_batch(self, X, batch_size=32, **kwargs):
        """
        :param X: `pd.Series` that contains raw text to featurize
        :param batch_size: int number of examples to process per batch
        :returns: list of np.ndarray representations of text
        """

        test_dataloader = self._to_dataloader(X)
                 
        features = []
        for batch in tqdm(test_dataloader, desc="Iteration"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask  = batch
            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            layers = np.concatenate(
                [
                    layer.detach().cpu().numpy() 
                    for i, layer in enumerate(all_encoder_layers) 
                    if i in range(1, 5)
                ], 
                axis=2
            )
            input_mask = np.expand_dims(input_mask, axis=2)
            embeddings = np.sum(layers * input_mask, axis=1) / np.sum(input_mask, axis=1)
            features.extend(embeddings)
        return np.vstack(features)
