import pandas as pd
import json
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


def crf_filter_text(text, crf):
    tokens = [(t.text, 'WORD') for t in tokenizer(text)]
    crf_feats = sent2features(tokens)
    crf_preds = crf.predict_single(crf_feats)
    text_out = ' '.join([tok[0] for tok, pred in zip(tokens, crf_preds) if pred != 'O'])
    return text_out


def word2features(sent, i, spacy_vectorizer=None):
    word = sent[i][0]
    postag = sent[i][1]
    # TODO: initialize spacy object here.



    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if spacy_vectorizer is not None:
        spacy_vector = spacy_vectorizer([word])[0]
        vector_features = {"vector_idx-"+str(i): j for i, j in enumerate(spacy_vector)}
        features.update(vector_features)
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
        if spacy_vectorizer is not None:
            spacy_vector = spacy_vectorizer([word1])[0]
            vector_features = {"-1:vector_idx-"+str(i): j for i, j in enumerate(spacy_vector)}
            features.update(vector_features)
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
        if spacy_vectorizer is not None:
            spacy_vector = spacy_vectorizer([word1])[0]
            vector_features = {"vector_idx-"+str(i): j for i, j in enumerate(spacy_vector)}
            features.update(vector_features)
    else:
        features['EOS'] = True

    return features


def sent2features(sent, spacy_vectorizer=None):
    return [word2features(sent, i, spacy_vectorizer) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def df_to_crf_feats(frame, labels=True):
    spans = get_spans(frame, trim_front=True)
    spantexts = [get_span_texts(span) for span in spans]
    features = [make_crf_features(spantext) for spantext in spantexts]
    X = [sent2features(s) for s in features]
    y = None
    if labels:
        y = [sent2labels(s) for s in features]
    return X, y

# CRF code ends here


# TODO: Rename all this 'span' nonsense

nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)


def strip_page(text):
    lines = text.split('\n')
    return '\n'.join(lines[:2])

# Note, will need too write a funtion that fixes the offsets
# Subtracting length of the two stripped lines

def get_spans_enso(X, Y, trim_front=False):
    spans = []
    for x, y in zip(X, Y):
        text = x
        offset = 0
        if trim_front:
            lines = text.split('\n')
            offset = len(lines[0]) + len(lines[1]) + 2
            text = '\n'.join(lines[2:])
        spans.append((text, sorted(
            [max(0, span[j] - offset) for span in y[0]
                for j in span if j in ['start', 'end']])))
    return spans

def get_spans(df, trim_front=False):
    spans = []
    for page in df.itertuples():
        text = page.text
        offset = 0
        if trim_front:
            lines = text.split('\n')
            offset = len(lines[0]) + len(lines[1]) + 2
            text = '\n'.join(lines[2:])
        spans.append((text, sorted(
            [max(0, span[j] - offset) for span in json.loads(page.question_1480)
                for j in span if j in ['startOffset', 'endOffset']])))
    return spans


def get_span_texts(span):
    span_texts = []
    # Add the text before any span occurs
    # Note, will be wanting to clip the first two lines
    # Script will be trimming negative information
    #for span in spans:
    page_texts = []
    numspans = len(span[1])
    # Note this condition will lead to _including_
    # pages with no rationales
    if numspans == 0:
        span_texts.append((span[0], "O"))
    if numspans > 0:
        span_texts.append((span[0][:span[1][0]], "O"))
    for i in range(numspans - 1): 
        text = span[0][span[1][i]:span[1][i+1]]
        label = "rationale" if (i % 2) == 0 else "O"
        span_texts.append((text, label))
    return span_texts



def make_crf_features(span_texts):
    features = []
    for span in span_texts:
        span_feats = []
        # Note, can iterate through tokenzer.pipe(...)
        # but would need to redo so we can match the labels up
        if span[1] == 'O':
            for token in tokenizer(span[0]):
                span_feats.append((token.text, "WORD", span[1]))
        else:
            for i, token in enumerate(tokenizer(span[0])):
                if i == 0:
                    span_feats.append((token.text, "WORD", 'B-'+span[1]))
                else:
                    span_feats.append((token.text, "WORD", 'I-'+span[1]))
        features += (span_feats)
    return features
