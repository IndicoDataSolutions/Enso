import os
import requests

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import json
import nltk
from nltk.tokenize import sent_tokenize

from enso import config
from enso.mode import ModeKeys

if __name__ == "__main__":
    nltk.download("punkt")
    task_type = ModeKeys.SEQUENCE
    filename = "Reuters-128.json"
    save_path = os.path.join(config.DATA_DIRECTORY, task_type.value, filename)
    if os.path.exists(save_path):
        print("{} already downloaded, skipping...".format(filename))
    else:
        url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
        r = requests.get(url)
        soup = bs(r.content.decode("utf-8"), "html.parser")
        docs = []
        for elem in soup.find_all("document"):
            single_entry = ["", []]
            for c in elem.find("textwithnamedentities").children:
                if type(c) == Tag:
                    sent_parts = sent_tokenize(c.text)
                    if len(sent_parts) == 1:
                        sent_parts = [c.text]

                    for i, text in enumerate(sent_parts):
                        if i == 1:
                            docs.append(single_entry)
                            single_entry = ["", []]

                        if c.name == "namedentityintext":
                            single_entry[1].append(
                                {
                                    "start": len(single_entry[0]),
                                    "end": len(single_entry[0]) + len(c.text),
                                    "label": "NAME",
                                }
                            )
                        single_entry[0] += text

            docs.append(single_entry)
        with open(save_path, "wt") as fp:
            json.dump(docs, fp, indent=1)
