import time
from numpy.lib.npyio import genfromtxt
from BinanceConnect import *
from datetime import datetime
import pickle
import os
import csv
from loggerSettings import logger
import chime
import pandas as pd
import numpy as np

class KlineData():
    def __init__(self):
        pass

    def getPickleFileName(self, pair, timeInterval, start, stop):
        return f"pickle//{pair}_{timeInterval}_{datetime.strptime(start, '%b %d, %Y').strftime('%m%d%Y')}_{datetime.strptime(stop, '%b %d, %Y').strftime('%m%d%Y')}.pkl"

    def getCSVFileName(self, pair, timeInterval, start, stop):
        return f"csv//{pair}_{timeInterval}_{datetime.strptime(start, '%b %d, %Y').strftime('%m%d%Y')}_{datetime.strptime(stop, '%b %d, %Y').strftime('%m%d%Y')}.csv"

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
        timeIntervals = ['1m']
        startDays = ['Jan 1, 2010']
        stop = 'Jul 1, 2021'
        dataDir = "E:\MarketData\CRYPTO\BINANCE"
        for pair in pairs:
            for timeInterval in timeIntervals:
                for start in startDays:
                    try:
                        # First pickle data/file
                        fileName = KlineData.getPickleFileName(self, pair, timeInterval, start, stop)
                        filePath = os.path.join(dataDir,fileName)
                        if not os.path.exists(filePath):
                            kLines = KlineData.fetchHistoricalData(self, pair, timeInterval, start, stop)
                            with open(filePath, 'wb') as f:
                                pickle.dump(kLines, f)
                            logger.debug(f"saved kline data to: {fileName}")
                        else:
                            logger.error(f"file: {fileName} already exists")
                        
                        # Second CSV data/file
                        fileNameCSV = KlineData.getCSVFileName(self, pair, timeInterval, start, stop)
                        filePathCSV = os.path.join(dataDir,fileNameCSV)
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
                        logger.error(f"failed to save kline data for {pair}, error: {e}")
            chime.play_tone(freq=1000, bursts=3, burst_time=0.1)

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

    def BinanceCSVtoDataframe(self, filename=None):
        """
        Load binance CSV file and 
        """
        try:
            if filename:
                if os.path.isfile(filename):
                    df = pd.read_csv(filename)
                    df = df.iloc[:, :-6]                                                # Drop last 6 columns
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']     # Set column names
                    df['date'] = pd.to_datetime(df['date'],unit='ms')                    # Convert unix to date time
                    df = df.set_index('date')                                                # Set the date as the index column
                    return df
                else:
                    raise Exception(f'No file: {filename}')
            else:
                raise Exception("No filename entered")
        except Exception as e:
            logger.error(e)

def calculateHeikenAshi(klines):
    """
    Take ohlc kline data and calcuate the heiken ashi candles for data and return updated dataframe
    """
    df = klines
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    df['ha_close']=(df['open'] + df['high'] + df['low'] + df['close'])/4
    idx = df.index.name
    df.reset_index(inplace=True)
    for i in range(0, len(df)):
        if i == 0:
            df.at[i, 'ha_open'] = (df.at[i, 'open'] + df.at[i, 'close']) / 2
        else:
            df.at[i, 'ha_open'] = (df.at[i - 1, 'ha_open'] + df.at[i - 1, 'ha_close']) / 2
    if idx:
        df.set_index(idx, inplace=True)
    df['ha_high']=df[['ha_open','ha_close','high']].max(axis=1)
    df['ha_low']=df[['ha_open','ha_close','low']].min(axis=1)
    
    return df   # Dataframe with both OHLC and HA candle data

def aggregateKlines(df, timeframe):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], downcast="float")

    # Aggregate data given timeframe
    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'ha_open': 'first', 'ha_high': 'max', 'ha_low': 'min', 'ha_close': 'last', 'volume': np.sum}
    df_new = df.resample(timeframe, offset=0).agg(ohlc).dropna()
    #df_new.fillna("backfill")
    return df_new       


import pathlib
if __name__ == "__main__":
    # dirPath = "F:\MarketData\STOCK\SP500\csv"
    # onlyfiles = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
    # for file in onlyfiles:
    #     # extension = pathlib.Path(file).suffix
    #     # split = file.split('.')
    #     # pair = split[0]
    #     # timeframe = '1d'
    #     if ' ' in file:
    #         newFileName = file.replace(" ", "")
    #         os.rename(os.path.join(dirPath, file), os.path.join(dirPath, newFileName))
    #         print(f"rename {file} to {newFileName}")

    dirPath = "F:\MarketData\CRYPTO\pickle"
    onlyfiles = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
    redo = ["BCHUSDT", "BNBUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT"]
    newonlyfiles = []
    for r in redo:
        for f in onlyfiles:
            if r in f and '_1m' in f:
                newonlyfiles.append(f)
                
    for file in newonlyfiles:
        filePath = os.path.join(dirPath, file)
        with open(filePath, 'rb') as f:
            df = pickle.load(f)
            df1m = calculateHeikenAshi(df)
        split = file.split('_')
        pair = split[0]
        timeframeTexts = ['5m', '10m', '15m', '30m', '1h', '2h', '4h', '1d', '1w']
        timeframes = ['5MIN', '10MIN', '15MIN', '30MIN', '1H', '2H', '4H', '1D', '1W']
        for i in range(len(timeframes)):
            timeframe = timeframes[i]
            timeframeText = timeframeTexts[i]
            newFileName = f"{pair}_{timeframeText}.pkl"
            newFilePath = os.path.join(dirPath, newFileName)
            df = aggregateKlines(df1m, timeframe) 
            with open(newFilePath, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved dataframe of new aggregated data for {timeframeText} to {newFilePath}")


        
    