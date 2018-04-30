"""
From: https://www.figure-eight.com/data-for-everyone/

Contributors looked at a single sentence and rated its emotional content based on Plutchik’s wheel of emotions. 18 emotional choices were presented to contributors for grading.
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/primary-plutchik-wheel-DFE.csv",
        text_column="sentence",
        target_column="emotion",
        filename="DetailedEmotion.csv"
    )
