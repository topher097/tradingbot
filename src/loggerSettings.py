import sys
import logging
from logging.config import dictConfig
import time

logging_config = dict(
    version=1,
    formatters={
        'verbose': {
            'format': ("[%(asctime)s] %(levelname)s "
                       "[%(name)s:%(lineno)s] %(message)s"),
            'datefmt': "%d/%b/%Y %H:%M:%S",
        },
        'simple': {
            'format': '%(levelname)s %(message)s',
        },
    },
    handlers={
        'api-logger': {'class': 'logging.handlers.RotatingFileHandler',
                           'formatter': 'verbose',
                           'level': 'DEBUG',
                           'filename': f'api_logs/api_{int(time.time())}.log',
                           'mode': 'w',
                           'maxBytes': 52428800,
                           'backupCount': 7},
        'plot-logger': {'class': 'logging.handlers.RotatingFileHandler',
                           'formatter': 'verbose',
                           'level': 'DEBUG',
                           'filename': f'plotter_logs/api_{int(time.time())}.log',
                           'mode': 'w',
                           'maxBytes': 52428800,
                           'backupCount': 7},
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple'}
    },
    loggers={
        'api_logger': {
            'handlers': ['api-logger', 'console'],
            'level': 'DEBUG',
        },
        'plot_logger': {
            'handlers': ['plot-logger', 'console'],
            'level': 'DEBUG'
        },
    }
)

dictConfig(logging_config)

logger = logging.getLogger('api_logger')
plotlogger = logging.getLogger('plot_logger')