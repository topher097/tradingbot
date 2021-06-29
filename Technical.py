from BinanceConnect import *
import talib
import numpy
from numpy import genfromtxt
from loggerSettings import logger

class Technical():
    def __init__(self):
        pass

    def getRSI(self, klines, type='close', timePeriod=30):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for RSI')
            return talib.RSI(d, timePeriod)
        except Exception as e:
            logger.error(e)

    def getParabolicSAR(self, klines, acceleration=0.02, maximum=0.2):
        try:           
            low = klines[:,1]
            high = klines[:,4]
            return talib.SAR(low, high, acceleration, maximum)
        except Exception as e:
           logger.error(e)