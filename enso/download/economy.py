"""
From: https://www.figure-eight.com/data-for-everyone/

Contributors looked at a single sentence and rated its emotional content based on Plutchik’s wheel of emotions. 18 emotional choices were presented to contributors for grading.
"""

from enso.download import generic_download


def convert_score_to_category(score):
    if 1 <= score <= 3:
        return "negative"
    elif 4 <= score <= 6:
        return "neutral"
    else:
        return "positive"


if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/us-economic-newspaper.csv",
        text_column="text",
        target_column="positivity",
        target_transformation=convert_score_to_category,
        filename="Economy.csv"
    )
