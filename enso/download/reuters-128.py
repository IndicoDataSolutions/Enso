import os
import requests

from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import json

from enso import config

if __name__ == "__main__":
    task_type = "SequenceLabeling"
    filename = "Reuters-128.json"
    save_path = os.path.join(config.DATA_DIRECTORY, task_type, filename)
    if os.path.exists(save_path):
        print("{} already downloaded, skipping...".format(filename))
        exit()
    url = "https://raw.githubusercontent.com/dice-group/n3-collection/master/reuters.xml"
    r = requests.get(url)
    soup = bs(r.content.decode("utf-8"), "html5lib")
    docs = []
    for elem in soup.find_all("document"):
        texts = []
        texts_no_labels = []
        just_labels = []

        single_entry = ["", []]
        char_loc = 0
        for c in elem.find("textwithnamedentities").children:
            if type(c) == Tag:
                if c.name == "namedentityintext":
                    single_entry[1].append(
                        {
                            "start": char_loc,
                            "end": char_loc + len(c.text),
                            "label": "NAME"
                        }
                    )

                single_entry[0] += c.text
                char_loc += len(c.text)

            docs.append(single_entry)
    with open(save_path, "wt") as fp:
        json.dump(docs, fp, indent=1)
