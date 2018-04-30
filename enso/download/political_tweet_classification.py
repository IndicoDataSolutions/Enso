"""
From: https://www.figure-eight.com/data-for-everyone/
Contributors looked at thousands of social media messages from US Senators and other American politicians to classify their content. Messages were broken down into audience (national or the tweeter’s constituency), bias (neutral/bipartisan, or biased/partisan), and finally tagged as the actual substance of the message itself (options ranged from informational, announcement of a media appearance, an attack on another candidate, etc.)
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/Political-media-DFE.csv",
        text_column="text",
        target_column="message",
        filename="PoliticalTweetClassification.csv"
    )

    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/Political-media-DFE.csv",
        text_column="text",
        target_column="bias",
        filename="PoliticalTweetBias.csv"
    )
