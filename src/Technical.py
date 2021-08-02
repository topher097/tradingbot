#from BinanceConnect import *
import talib
import numpy as np
import pandas as pd
from numpy import dtype, genfromtxt
from loggerSettings import logger

class Technical():
    def __init__(self):
        pass
    
    """ ********** OVERLAP STUDIES ********** """
    def getParabolicSAR(self, klines, attributes):
        # type='close', timePeriod=30
        try:         
            acceleration    = attributes['acceleration']
            maximum         = attributes['maximum']  
            low             = klines['low'].to_numpy(dtype='f8')
            high            = klines['high'].to_numpy(dtype='f8')
            return talib.SAR(low, high, acceleration=acceleration, maximum=maximum)
        except Exception as e:
            logger.error(e)

    def getSMA(self, klines, attributes):
        # type='close', timePeriod=30
        try:           
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']
            d           = klines[type].to_numpy(dtype='f8')
            return talib.SMA(d, timeperiod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getEMA(self, klines, attributes):
        # type='close', timePeriod=30
        try:           
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']
            d           = klines[type].to_numpy(dtype='f8') 
            return talib.EMA(d, timeperiod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getMA(self, klines, attributes):
        # type='close', timePeriod=30
        try:           
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']
            d           = klines[type].to_numpy(dtype='f8')
            return talib.MA(d, timeperiod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getBollingerBands(self, klines, attributes):
        # ex: type='close', timePeriod=30, stdevup=2, stdevdown=2, matype=0
        try:  
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']
            stdevup     = attributes['stdevpup']
            stdevdown   = attributes['stdevpdown']         
            matype      = attributes['matype']
            d           = klines[type].to_numpy(dtype='f8')
            upperBand, middleBand, lowerBand = talib.BBANDS(d, timeperiod=timePeriod, nbdevup=stdevup, nbdevdn=stdevdown, matype=matype)
            return upperBand, middleBand, lowerBand
        except Exception as e:
            logger.error(e)

    """ ********** MOMENTUM INDICATORS ********** """
    def getRSI(self, klines, attributes):
        # type='close', timePeriod=30
        try:      
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']     
            d           = klines[type].to_numpy(dtype='f8')
            return talib.RSI(d, timeperiod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getROC(self, klines, attributes):
        # type='close', timePeriod=30
        try:      
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']     
            d           = klines[type].to_numpy(dtype='f8')
            return talib.ROC(d, timeperiod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getMACD(self, klines, attributes):
        # type='close', fastPeriod=15, fastMAtype=0, slowPeriod=30, slowMAtype=0, signalPeriod=9, signalMAtype=0
        try:           
            type            = attributes['type']
            fastPeriod      = attributes['fastPeriod']
            fastMAtype      = attributes['fastMAtype']
            slowPeriod      = attributes['slowPeriod']
            slowMAtype      = attributes['slowMAtype']
            signalPeriod    = attributes['signalPeriod']
            signalMAtype    = attributes['signalMAtype']
            d               = klines[type].to_numpy(dtype='f8')
            macd, macdsignal, macdhist = talib.MACDEXT(d, fastperiod=fastPeriod, fastmatype=fastMAtype, slowperiod=slowPeriod, slowmatype=slowMAtype, signalperiod=signalPeriod, signalmatype=signalMAtype)
            return macd, macdsignal, macdhist
        except Exception as e:
            logger.error(e)

    def getMFI(self, klines, attributes):
        # type='close', timePeriod=30
        try:           
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']     
            d           = klines[type].to_numpy(dtype='f8')
            high        = klines['high'].to_nump(dtype='f8')
            low         = klines['low'].to_nump(dtype='f8')
            volume      = klines['volume'].to_nump(dtype='f8')
            return talib.MFI(high, low, d, volume, timeperiod=timePeriod)
        except Exception as e:
            logger.error(e)

    def getMOM(self, klines, attributes):
        # type='close', timePeriod=30
        try:      
            type        = attributes['type']
            timePeriod  = attributes['timePeriod']     
            d           = klines[type].to_numpy(dtype='f8')
            return talib.MOM(d, timeperiod=timePeriod)
        except Exception as e:
            logger.error(e)