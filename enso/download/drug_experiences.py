"""
From: https://www.figure-eight.com/data-for-everyone/

Contributors read color-coded sentences and determined what the relationship of a drug was to certain symptoms or diseases. There are two types of relationships. A drug either:

Caused side effects – [Drug] gave me [symptom]
Was effective against a condition – [Drug] helped my [disease]
Is prescribed for a certain disease – [Drug] was given to help my [disease]
Is contraindicated in – [Drug] should not be taken if you have [disease or symptom]
The second similarity was more about the statement itself. Those broke down into:

Personal experiences – I started [drug] for [disease]
Personal experiences negated – [Drug] did not cause [symptom]
Impersonal experiences – I’ve heard [drug] causes [symptom]
Impersonal experiences negated – I’ve read [drug] doesn’t cause [symptom]
Question – Have you tried [drug]?
"""
from enso.download import generic_download, html_to_text


if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/drug-relation-dfe.csv",
        text_column="text",
        target_column="human_relation",
        text_transformation=html_to_text,
        filename="DrugReviewType.csv"
    )

    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/drug-relation-dfe.csv",
        text_column="text",
        target_column="human_relation_type",
        text_transformation=html_to_text,
        filename="DrugReviewIntent.csv"
    )
