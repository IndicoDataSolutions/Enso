"""
From: https://www.figure-eight.com/data-for-everyone/
A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as “late flight” or “rude service”).
"""

from enso.download import generic_download


if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/Airline-Sentiment-2-w-AA.csv",
        text_column="text",
        target_column="negativereason",
        filename="AirlineNegativity.csv"
    )
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/Airline-Sentiment-2-w-AA.csv",
        text_column="text",
        target_column="airline_sentiment",
        filename="AirlineSentiment.csv"
    )
