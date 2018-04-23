"""
From: https://www.figure-eight.com/data-for-everyone/

Contributors looked at over 10,000 tweets culled with a variety of searches like “ablaze”, “quarantine”, and “pandemonium”, then noted whether the tweet referred to a disaster event (as opposed to a joke with the word or a movie review or something non-disastrous).
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/socialmedia-disaster-tweets-DFE.csv",
        text_column="text",
        target_column="choose_one",
        filename="SocialMediaDisasters.csv"
    )
