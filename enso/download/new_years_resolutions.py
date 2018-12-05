"""
From: https://www.figure-eight.com/data-for-everyone/
A Twitter topic analysis of users’ 2015 New Year’s resolutions.
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/New-years-resolutions-DFE.csv",
        text_column="text",
        target_column="Resolution_Category",
        filename="NewYearsResolutions.csv"
    )
