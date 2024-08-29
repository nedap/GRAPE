import logging
from typing import Union


logger_initialized: dict[str, bool] = {}

def get_logger(name: str, log_level: int=logging.INFO) -> logging.Logger:
    """
    Creates a simple logger, to log information to the console. 
    :param name: The name of the logger. 
    :param log_level: INFO, ERROR etc.
    :returns: Logger.
    """

    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    
    # Prevents reinitializing already initialized loggers, especially in hierarchical logger structures.
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    
    # Adjust log level of existing StreamHandlers to avoid duplicate logging in certain environments.
    for handler in logger.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]


    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Assign formatter and log level to each handler and attach them to the logger.
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger_initialized[name] = True

    return logger


def print_log(msg: str, logger: Union[None, str]=None, level: int=logging.INFO):
    """
    Print a log message either to the console, to a specified logger, or silently.
    
    :param msg: The message to be logged.
    :param 
    """
    if logger is None:
        # Print the message to the console if no logger is specified.
        print(msg)
    elif isinstance(logger, logging.Logger):
        # If a Logger object is provided, use it to log the message.
        logger.log(level=level, msg=msg)
    elif logger == 'silent':
        # If 'silent' is specified, do nothing (i.e., silently ignore the message).
        pass
    elif isinstance(logger, str):
        # If a string is provided, treat it as the name of a logger to fetch and use.
        _logger = get_logger(logger)
        _logger.log(level=level, msg=msg)
    else:
        # Raise an error if the `logger` argument is not one of the expected types.
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
