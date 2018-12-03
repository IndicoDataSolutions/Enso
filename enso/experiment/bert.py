import os
import json
import logging

import pandas as pd
import torch
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from indicoio.custom import Collection
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertModel, BertForMaskedLM, BertAdam

from enso.experiment import ClassificationExperiment
from enso.config import RESULTS_DIRECTORY
from enso.registry import Registry, ModeKeys


logger = logging.getLogger(__file__)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



class Example:
    """A single text + label pair"""

    def __init__(self, text, label):
        self.text = text
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, seq_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seq_ids = seq_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for example in examples:
        tokens = ["[CLS]"] + tokenizer.tokenize(example.text) + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        seq_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            seq_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(seq_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                seq_ids=seq_ids,
                input_mask=input_mask,
                label_id=label_id
            )
        )
    return features


@Registry.register_experiment(ModeKeys.CLASSIFY, requirements=[("Featurizer", "PlainTextFeaturizer")])
class BERT(ClassificationExperiment):
    """
    LanguageModel finetuning as an alternative to simple models trained on top of pretrained features.
    """

    param_grid = {}

    def __init__(self, *args, **kwargs):
        """Initialize internal classifier."""
        super().__init__(*args, **kwargs)
        self.model = None

    def _to_dataloader(self, X, y, labels, random_sample=False):
        examples = [Example(text=x, label=label) for x, label in zip(X, y)]
        features = convert_examples_to_features(
            examples, labels, self.max_seq_length, tokenizer
        )

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_seq_ids = torch.tensor([f.seq_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_seq_ids, all_input_mask, all_label_ids)
        if random_sample:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size)

    def fit(self, X, y):
        """
        :param X: `np.ndarray` of raw text sampled from training data.
        :param y: `np.ndarray` of corresponding targets sampled from training data.
        """
        
        ############
        # SETTINGS #
        ############
        self.batch_size = 2
        self.num_train_epochs = 3
        self.max_seq_length = 512
        ############
        
        self.num_train_steps = int(
            (len(X) / self.batch_size) * self.num_train_epochs
        )
        self.device = 'cuda'
        self.labels = list(np.unique(y))
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
            num_labels=len(self.labels)
        )
        self.model.to(self.device)
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=5e-5,
            warmup=0.1,
            t_total=self.num_train_steps
        )
        
        self.model.train()

        train_dataloader = self._to_dataloader(X, y, labels=self.labels, random_sample=True)
        for _ in trange(self.num_train_epochs, desc="Epoch"):
            for batch in tqdm(train_dataloader, desc="Iteration"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, seq_ids, input_mask, label_ids = batch
                loss = self.model(input_ids, seq_ids, input_mask, label_ids)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

    def predict(self, X, **kwargs):
        """Predict results on test set based on current internal model."""
        self.model.eval()

        target_placeholder = [self.labels[0]] * len(X)
        test_dataloader = self._to_dataloader(X, target_placeholder, labels=self.labels, random_sample=True)
        preds = []
        for batch in tqdm(test_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, seq_ids, input_mask, _ = batch
            with torch.no_grad():
                logits = self.model(input_ids, seq_ids, input_mask)
                probs = torch.nn.Softmax(dim=-1)(logits)
            probs = probs.detach().cpu().numpy()
            batch_preds = [dict(zip(self.labels, prob_arr)) for prob_arr in probs]
            preds.extend(batch_preds)

        return pd.DataFrame.from_records(preds)

    def cleanup(self):
        del self.model
