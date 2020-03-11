import os
import json
import re

import nltk
from nltk.corpus import brown

from tqdm import tqdm

from enso import config
from enso.mode import ModeKeys

only_tags_with = "N"


def label_preproc(label, regex, whole_tag=False):
    for sublabel in reversed(re.split("[+,-]", label)):
        if re.match(regex, sublabel):
            if whole_tag:
                return label
            return sublabel
    return None


def brown_corpus_tags(task_name, tag_regex, whole_tag=False):
    task_type = ModeKeys.SEQUENCE
    filename = "brown_{}.json".format(task_name)

    task_dir = os.path.join(config.DATA_DIRECTORY, task_type.value,)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_path = os.path.join(task_dir, filename)

    docs = []
    for tagged_sent in tqdm(brown.tagged_sents(), desc=task_name):
        doc_text = ""
        doc_annotations = []
        last_label = []
        for sub_str, label in tagged_sent:
            label = label_preproc(label, tag_regex, whole_tag)

            if doc_text:
                doc_text += " "

            doc_location = len(doc_text)
            doc_text += sub_str
            doc_end = len(doc_text)

            if doc_annotations and label is not None and label == last_label:
                doc_annotations[-1]["end"] = doc_end

            elif label is not None:
                doc_annotations.append(
                    {
                        "start": doc_location,
                        "end": doc_location + len(sub_str),
                        "label": label
                    }
                )

            last_label = label
        docs.append([doc_text, doc_annotations])

    with open(save_path, "wt") as fp:
        json.dump(docs, fp, indent=1)


if __name__ == "__main__":
    nltk.download("brown")
    brown_tasks = [
        ("nouns", r'^N[A-Z]*', False),
        ("verbs", r'^V[A-Z]*', False),
        ("adverbs", r'^R[A-Z]*', False),
        ("pronouns", r'^P[A-Z]*', False),
        ("all", r'[A-Z]*', False)
    ]

    for task in brown_tasks:
        brown_corpus_tags(*task)
