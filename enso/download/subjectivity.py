"""
From: https://www.figure-eight.com/data-for-everyone/

Subjectivity task from SentEval.  Note that performance on this dataset is not comparable to official SentEval scores because of differences in data splitting.
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://s3.amazonaws.com/enso-data/Subjectivity.csv",
        text_column="Text",
        target_column="Target",
        filename="Subjectivity.csv"
    )
