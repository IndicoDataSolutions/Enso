"""
From: https://www.figure-eight.com/data-for-everyone/
A data categorization job concerning what corporations actually talk about on social media. Contributors were asked to classify statements as information (objective statements about the company or it’s activities), dialog (replies to users, etc.), or action (messages that ask for votes or ask users to click on links, etc.).
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/Airline-Sentiment-2-w-AA.csv",
        text_column="text",
        target_column="negativereason",
        filename="AirlineNegativity.csv"
    )
