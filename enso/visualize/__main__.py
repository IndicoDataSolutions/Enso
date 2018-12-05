import logging

from enso.config import EXPERIMENT_NAME
from enso.visualize import Visualization

if __name__ == "__main__":
    logging.info('Loading Results...')
    visualization = Visualization(EXPERIMENT_NAME)
    logging.info('Painting Pretty Pictures...')
    visualization.visualize()
