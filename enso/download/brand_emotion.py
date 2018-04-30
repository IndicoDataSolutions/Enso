"""
From: https://www.figure-eight.com/data-for-everyone/
Contributors evaluated tweets about multiple brands and products. The crowd was asked if the tweet expressed positive, negative, or no emotion towards a brand and/or product. If some emotion was expressed they were also asked to say which brand or product was the target of that emotion.
"""

from enso.download import generic_download


if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/judge-1377884607_tweet_product_company.csv",
        text_column="tweet_text",
        target_column="is_there_an_emotion_directed_at_a_brand_or_product",
        filename="BrandEmotion.csv"
    )
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/judge-1377884607_tweet_product_company.csv",
        text_column="tweet_text",
        target_column="emotion_in_tweet_is_directed_at",
        filename="BrandEmotionCause.csv"
    )
