import logging
import os

def logger_init(name):
    logger = logging.getLogger('name')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(filename)s:%(levelname)s:%(lineno)d:%(message)s')

    if not os.path.exists('logger'):
        os.makedirs('logger')
    else:
        file_handler = logging.FileHandler('logger/tmr_fastapi.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger