import logging
import psutil

from mpi4py import MPI

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
            result = 'process {}'.format(MPI.COMM_WORLD.Get_rank())
        else:
            result = self.__dict__.get(name, '?')
        return result

    def __iter__(self):
        """
        To allow iteration over keys, which will be merged into
        the LogRecord dict before formatting and output.
        """
        keys = ['memory', 'rank']
        keys.extend(self.__dict__.keys())
        return keys.__iter__()

    def _format_free_memory(self):
        BYTES_PER_MB = 2**20
        mem_info = psutil.virtual_memory()
        free_mb = mem_info.available / BYTES_PER_MB
        total_mb = mem_info.total / BYTES_PER_MB
        return '{}/{}MB'.format(free_mb, total_mb)
        
LOG_OUTPUT_DIR = None
ADD_EXTRA_LOGGING_INFO = True
def get_logger(name, log_level):
    logger = logging.getLogger(name)

    # create console handler
    if len(logger.handlers)==0:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        # create formatter and add it to the handlers
        if ADD_EXTRA_LOGGING_INFO:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(rank)s - %(memory)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)
        
        if LOG_OUTPUT_DIR is not None:
            hdlr = logging.FileHandler('{}/output.log'.format(LOG_OUTPUT_DIR))
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr) 
    else:
        logger.handlers[0].setLevel(log_level)
        
    # Wrap logger in an adapter to add memory
    if ADD_EXTRA_LOGGING_INFO:
        logger = logging.LoggerAdapter(logger, LoggerInfo())
    
    logger.setLevel(log_level)
    return logger
    
    
    
    
    
    
    
    
    
    
    
