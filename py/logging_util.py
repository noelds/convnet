import os
import sys
import logging


def initialize_logger(name, output_dir, level=logging.DEBUG,
                      formatting="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(formatting)

    # create console handler and set level to info
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "error.log"), "w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, "all.log"), "w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
