"""
From: https://www.figure-eight.com/data-for-everyone/

Contributors viewed tweets regarding a variety of left-leaning issues like legalization of abortion, feminism, Hillary Clinton, etc. They then classified if the tweets in question were for, against, or neutral on the issue (with an option for none of the above). After this, they further classified each statement as to whether they expressed a subjective opinion or gave facts.
"""
from enso.download import generic_download


if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/progressive-tweet-sentiment.csv",
        text_column="tweet",
        target_column="q1_from_reading_the_tweet_which_of_the_options_below_is_most_likely_to_be_true_about_the_stance_or_outlook_of_the_tweeter_towards_the_target",
        filename="PoliticalTweetAlignment.csv"
    )

    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/progressive-tweet-sentiment.csv",
        text_column="tweet",
        target_column="q2_which_of_the_options_below_is_true_about_the_opinion_in_the_tweet",
        filename="PoliticalTweetSubjectivity.csv"
    )

    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/progressive-tweet-sentiment.csv",
        text_column="tweet",
        target_column="target",
        filename="PoliticalTweetTarget.csv"
    )
