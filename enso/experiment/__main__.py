import logging

from enso.config import EXPERIMENT_NAME

from . import Experimentation

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Experimentation Started...")
    experimentation = Experimentation(EXPERIMENT_NAME)
    logging.info("Running Experiments...")
    experimentation.run_experiments()
