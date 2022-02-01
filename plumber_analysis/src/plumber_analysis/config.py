"""
Logging utilities for Plumber.
"""

import logging

def set_plumber_logging(level):
    logger = logging.getLogger("plumber_analysis_lib")
    logger.setLevel(level)

def set_global_logging(level=None):
    if level is None:
        level = logging.INFO
    logging.basicConfig(
            level=level,
            filename="plumber.log",
            filemode="w",
            format="%(name)s - $(levelname)s - $(message)s")

def enable_compat_logging():
    logging.basicConfig(level=logging.INFO,
                        filename='plumber.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')
