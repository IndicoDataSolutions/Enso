"""
From: https://www.figure-eight.com/data-for-everyone/

Contributors read sentences in which both a chemical (like Aspirin) and a disease (or side-effect) were present. They then determined if the chemical directly contributed to the disease or caused it. Dataset includes chemical names, disease name, and aggregated judgments of five (as opposed to the usual three) contributors.
"""

from enso.download import generic_download

if __name__ == "__main__":
    generic_download(
        url="https://www.figure-eight.com/wp-content/uploads/2016/03/chemicals-and-disease-DFE.csv",
        text_column="form_sentence",
        target_column="verify_relationship",
        filename="ChemicalDiseaseCauses.csv"
    )
