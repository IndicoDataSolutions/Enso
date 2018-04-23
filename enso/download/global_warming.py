"""
From: https://www.figure-eight.com/data-for-everyone/
Contributors evaluated tweets for belief in the existence of global warming or climate change. The possible answers were “Yes” if the tweet suggests global warming is occurring, “No” if the tweet suggests global warming is not occurring, and “I can’t tell” if the tweet is ambiguous or unrelated to global warming. We also provide a confidence score for the classification of each tweet.
"""

from enso.download import generic_download

def words_to_char(val):
    conversion = {
        "Yes": 'Y',
        "No": 'N'
    }
    converted_val = conversion.get(val, val)
    return converted_val


if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/1377884570_tweet_global_warming.csv",
        text_column="tweet",
        target_column="existence",
        target_transformation=words_to_char,
        filename="GlobalWarming.csv"
    )
