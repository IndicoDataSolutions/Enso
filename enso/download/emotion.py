"""
From: https://www.figure-eight.com/data-for-everyone/

In a variation on the popular task of sentiment analysis, this dataset contains labels for the emotional content (such as happiness, sadness, and anger) of texts. Hundreds to thousands of examples across 13 labels.
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/07/text_emotion.csv",
        text_column="content",
        target_column="sentiment",
        filename="Emotion.csv"
    )
