from BinanceConnect import *
import talib
import numpy
from numpy import genfromtxt
from loggerSettings import logger

class Technical():
    def __init__(self):
        pass

    """ ********** OVERLAP STUDIES ********** """
    def getParabolicSAR(self, klines, acceleration=0.02, maximum=0.2):
        try:           
            low = klines[:,3]
            high = klines[:,2]
            return talib.SAR(low, high, acceleration=acceleration, maximum=maximum)
        except Exception as e:
           logger.error(e)

    def getSMA(self, klines, type='close', timePeriod=30):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for SMA')
            return talib.SMA(d, timePeriod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getEMA(self, klines, type='close', timePeriod=30):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for EMA')
            return talib.EMA(d, timePeriod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getMA(self, klines, type='close', timePeriod=30):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for MA')
            return talib.MA(d, timePeriod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getBollingerBands(self, klines, type='close', timePeriod=30, stdevup=2, stdevdown=2, matype=0):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for Bollinger Bands')
            upperBand, middleBand, lowerBand = talib.BBANDS(d, timePeriod=timePeriod, nbdevup=stdevup, nbdevdn=stdevdown, matype=0)
            return upperBand, middleBand, lowerBand
        except Exception as e:
            logger.error(e)

    """ ********** MOMENTUM INDICATORS ********** """
    def getRSI(self, klines, type='close', timePeriod=30):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for RSI')
            return talib.RSI(d, timePeriod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getMACD(self, klines, type='close', fastPeriod=15, fastMAtype=0, slowPeriod=30, slowMAtype=0, signalPeriod=9, signalMAtype=0):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for MACD')
            macd, macdsignal, macdhist = talib.MACDEXT(d, fastperiod=fastPeriod, fastmatype=fastMAtype, slowperiod=slowPeriod, slowmatype=slowMAtype, signalperiod=signalPeriod, signalmatype=signalMAtype)
            return macd, macdsignal, macdhist
        except Exception as e:
            logger.error(e)

    def getMFI(self, klines, type='close', timePeriod=30):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for MFI')
            high    = klines[:,2]
            low     = klines[:,3]
            volume  = klines[:,5]
            return talib.MFI(high, low, d, volume, timePeriod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getMOM(self, klines, type='close', timePeriod=30):
        try:           
            if type=='open':
                d = klines[:,1]
            elif type=='close':
                d = klines[:,4]
            else:
                raise Exception('Illegal type for MOM')
            return talib.MOM(d, timePeriod=timePeriod)
        except Exception as e:
            logger.error(e)