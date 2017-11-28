import logging

from enso.visualize import Visualization

if __name__ == "__main__":
    logging.info('Loading Results...')
    visualization = Visualization()
    logging.info('Painting Pretty Pictures...')
    visualization.visualize()
