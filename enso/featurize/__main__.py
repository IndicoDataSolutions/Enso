import logging

from . import Featurization

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing Featurizers...")
    featurization = Featurization()
    logging.info("Converting Datasets...")
    featurization.run()
