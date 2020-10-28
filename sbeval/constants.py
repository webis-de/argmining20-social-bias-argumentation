import logging
import sys

from os import path

# Basic logging configuration
LOGGING_CONFIG = {
    "stream": sys.stdout,
    "format": "%(levelname)s:%(asctime)s:%(message)s",
    "level": logging.INFO,
    "datefmt": "%Y-%m-%d %H:%M:%S"}

# Delta tolerance to the original weat test results
WEAT_TEST_TOLERANCE = 0.03

# Directory containing the pre-trained word vector files
WORD_VECTOR_DIR = path.join("word_vectors")
