import os.path
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

from enso import config
from enso.mode import ModeKeys


def generic_download(
    url,
    text_column,
    target_column,
    filename,
    save=True,
    task_type=ModeKeys.CLASSIFY,
    text_transformation=None,
    target_transformation=None,
):
    task_dir = os.path.join(config.DATA_DIRECTORY, task_type.value)
    save_path = os.path.join(task_dir, filename)
    if os.path.exists(save_path):
        print("{} already downloaded, skipping...".format(filename))
        return

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    response = requests.get(url)
    _file = StringIO(response.text.replace("\r", "\n"))
    df = pd.read_csv(_file)
    df = df.dropna(subset=[text_column, target_column])

    new_df = pd.DataFrame(columns=["Text", "Target"])
    new_df["Text"], new_df["Target"] = df[text_column], df[target_column]

    if text_transformation is not None:
        new_df["Text"] = new_df["Text"].apply(text_transformation)
    if target_transformation is not None:
        new_df["Target"] = new_df["Target"].apply(target_transformation)

    if save:
        new_df.to_csv(save_path, index=False)

    return new_df


def html_to_text(text):
    return BeautifulSoup(text, "html.parser").get_text()
