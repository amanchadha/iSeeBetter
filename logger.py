"""
logger.py - Logging functionality for iSeeBetter.

Aman Chadha | aman@amanchadha.com
"""

import logging, sys

# String format used for logging
LOG_FORMAT_STRING       = "[%(levelname)8.8s] %(message)s"
LOG_FORMAT_STRING_DEBUG = "[%(levelname)8.8s][%(filename)s:%(lineno)s:%(funcName)15s()] %(message)s"

logger = logging.getLogger('root')
debug = logger.debug
info = logger.info
warning = logger.warning

def initLogger(debugFlag):
    # Set logger level
    logging.basicConfig(format=LOG_FORMAT_STRING_DEBUG if debugFlag else LOG_FORMAT_STRING,
                        level=logging.DEBUG if debugFlag else logging.INFO)

def errorOut(errorString, e=None):
    logging.error(errorString + (' Error returned was: '+str(e) if e is not None else ''))
    sys.exit(1)