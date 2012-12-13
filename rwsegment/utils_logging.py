import logging
import os

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING 
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL


class LoggerInfo:
    """
    A class for adding extra info to loggers
    """

    def __getitem__(self, name):
        """
        To allow this instance to look like a dict.
        """
        if name == 'memory':
            result = self._format_free_memory()
        elif name == 'rank':
            result = 'process {}'.format(RANK)
        elif name=='host':
            result = 'host {}'.format(HOST)
        else:
            result = self.__dict__.get(name, '?')
        return result

    def __iter__(self):
        """
        To allow iteration over keys, which will be merged into
        the LogRecord dict before formatting and output.
        """
        keys = ['memory', 'rank', 'host']
        keys.extend(self.__dict__.keys())
        return keys.__iter__()

    def __init__(self):
        #try:
        if 1:
            import psutil
            self._format_free_memory = self._psutil_format_free_memory
        #except ImportError:
        else:
            self._format_free_memory = self._no_psutil_format_free_memory
    
    def _no_psutil_format_free_memory(self):
            return ''
    
    def _psutil_format_free_memory(self):
        import psutil
        BYTES_PER_MB = 2**20
        mem_info = psutil.virtual_memory()
        free_mb = mem_info.available / BYTES_PER_MB
        total_mb = mem_info.total / BYTES_PER_MB
        return '{}/{}MB'.format(free_mb, total_mb)
    
            
        
LOG_OUTPUT_DIR = None
ADD_EXTRA_LOGGING_INFO = True

import mpi
RANK = mpi.RANK
HOST = mpi.HOST

def get_logger(name, log_level):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # create console handler
    if len(logger.handlers)==0:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        # create formatter and add it to the handlers
        if ADD_EXTRA_LOGGING_INFO:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(rank)s - %(host)s - %(memory)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)
        
        if LOG_OUTPUT_DIR is not None:
            
            ## Make sure that log directory exists
            if RANK == 0:
                if not os.path.exists(LOG_OUTPUT_DIR):
                    os.makedirs(LOG_OUTPUT_DIR)
            else:
                while not os.path.isdir(LOG_OUTPUT_DIR):
                    pass
            
            hdlr = logging.FileHandler('{}/output{}.log'.format(LOG_OUTPUT_DIR, RANK))
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr) 
    else:
        logger.handlers[0].setLevel(log_level)
        
    # Wrap logger in an adapter to add memory
    if ADD_EXTRA_LOGGING_INFO:
        logger = logging.LoggerAdapter(logger, LoggerInfo())
    
    return logger
    
    
    
    
    
    
    
    
    
    
    
