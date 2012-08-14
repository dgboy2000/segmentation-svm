import logging

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING 
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL


def get_logger(name, log_level):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # create console handler with a higher log level
    if len(logger.handlers)==0:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)
    else:
        logger.handlers[0].setLevel(log_level)
    
    return logger
