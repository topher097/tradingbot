from numpy.lib.npyio import genfromtxt
from BinanceConnect import *
from datetime import datetime
import pickle
import os
import csv
from loggerSettings import logger

class KlineData():
    def __init__(self):
        pass

    def getPickleFileName(self, pair, timeInterval, start, stop):
        return f"historical_data\\{pair}_{timeInterval}_{datetime.strptime(start, '%b %d, %Y').strftime('%m%d%Y')}_{datetime.strptime(stop, '%b %d, %Y').strftime('%m%d%Y')}.pkl"

    def getCSVFileName(self, pair, timeInterval, start, stop):
        return f"historical_data_csv\\{pair}_{timeInterval}_{datetime.strptime(start, '%b %d, %Y').strftime('%m%d%Y')}_{datetime.strptime(stop, '%b %d, %Y').strftime('%m%d%Y')}.csv"

    def fetchHistoricalData(self, pair, timeInterval='30m', start='Jan 1, 2018', stop=datetime.utcnow().strftime('%b %d, %Y')):
        logger.debug(f"Pulling kline data for {pair} with interval of {timeInterval}, starting {start} and ending {stop}...")
        return self.client.get_historical_klines(pair, timeInterval, start, stop)

    def fetchCurrentData(self, pair, timeInterval='30m'):
        try:
            logger.debug(f"Pulling kline data for {pair} with interval of {timeInterval}...")
            return self.client.get_klines(pair, timeInterval)
        except Exception as e:
            logger.error(f"Failed retrieving current kLine data for {pair} with time interval of {timeInterval}")

    def saveHistoricalData(self, pairs):
        timeIntervals = ['1d', '4h', '2h', '1h', '30m', '5m', '1m']
        startDays = ['Jan 1, 2017', 'Jan 1, 2018', 'Jan 1, 2019', 'Jan 1, 2020', 'Jan 1, 2021']
        stop = 'Jun 1, 2021'
        for pair in pairs:
            for timeInterval in timeIntervals:
                for start in startDays:
                    try:
                        # First pickle data/file
                        fileName = KlineData.getPickleFileName(self, pair, timeInterval, start, stop)
                        filePath = os.path.join(self.BASEDIR,fileName)
                        if not os.path.exists(filePath):
                            kLines = KlineData.fetchHistoricalData(self, pair, timeInterval, start, stop)
                            with open(filePath, 'wb') as f:
                                pickle.dump(kLines, f)
                            logger.debug(f"saved kline data to: {fileName}")
                        else:
                            logger.error(f"file: {fileName} already exists")
                        
                        # Second CSV data/file
                        fileNameCSV = KlineData.getCSVFileName(self, pair, timeInterval, start, stop)
                        filePathCSV = os.path.join(self.BASEDIR,fileNameCSV)
                        if not os.path.exists(filePathCSV):
                            with open(filePath, 'rb') as f:
                                kLines = pickle.load(f)
                            with open(filePathCSV, 'w', newline='') as csvf:
                                candlestick_writer = csv.writer(csvf, delimiter=',')
                                candlestick_writer.writerows(kLines)
                            logger.debug(f"saved kline data csv to: {fileNameCSV}")
                        else:
                            logger.debug(f"csv file: {fileNameCSV} already exists")
                    except Exception as e:
                        logger.error(f"failed to save kline data, error: {e}")

    def loadHistoricalPickleData(self, pair, timeInterval, start, stop):
        try:
            print(f"Fetching historical pickle kLine data for {pair}")
            fileName = KlineData.getPickleFileName(self, pair, timeInterval, start, stop)
            filePath = os.path.join(self.BASEDIR,fileName)
            if not os.path.exists(filePath):
                kLines = KlineData.fetchHistoricalData(self, pair, timeInterval, start, stop)
                with open(filePath, 'wb') as f:
                    pickle.dump(kLines, f)
            else:
                with open(filePath, 'rb') as f:
                    kLines = pickle.load(f)
            logger.info(f"Successfully fetched kLine pickle historical data for {pair}")
            return kLines
        except Exception as e:
            logger.error(f"Unable to fetch kLine historical pickle data for {pair} due to error: {e}")

    def loadHistoricalCSVData(self, pair, timeInterval, start, stop):
        try:
            logger.debug(f"Fetching historical CSV kLine data for {pair}")
            fileNameCSV = KlineData.getCSVFileName(self, pair, timeInterval, start, stop)
            filePathCSV = os.path.join(self.BASEDIR,fileNameCSV)
            if not os.path.exists(filePathCSV):
                fileNamePkl = KlineData.getPickleFileName(self, pair, timeInterval, start, stop)
                filePathPkl = os.path.join(self.BASEDIR,fileNamePkl)
                if not os.path.exists(filePathPkl):
                    kLines = KlineData.fetchHistoricalData(self, pair, timeInterval, start, stop)
                    with open(filePathPkl, 'wb') as f:
                        pickle.dump(kLines, f)
                else:
                    with open(filePathPkl, 'rb') as f:
                        kLines = pickle.load(f)
                with open(filePathCSV, 'w', newline='') as csvf:
                    candlestick_writer = csv.writer(csvf, delimiter=',')
                    candlestick_writer.writerows(kLines)
                logger.debug(f"Saved kline data csv to: {fileNameCSV}")
            kLines = genfromtxt(fileNameCSV, delimiter=',')
            logger.info(f"Successfully fetched kLine CSV historical data for {pair} as numpy array")
            return kLines
        except Exception as e:
            logger.error(f"Unable to fetch kLine historical CSV data for {pair} due to error: {e}")


    