import logging


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_hander = logging.FileHandler(f'{logger_name}.log', mode='w')
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s'
    )
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)
    return logger
