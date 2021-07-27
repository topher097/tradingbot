"""
Using findatapy to download data for many assets, stored in folder specified
"""

from findatapy.util import SwimPool
from requests.sessions import default_hooks; SwimPool()
from findatapy.market import Market, MarketDataRequest, MarketDataGenerator
market = Market(market_data_generator=MarketDataGenerator())
from loggerSettings import logger
import os
import pickle

class CryptoHistoricalData():

    def __init__(self, config=None) -> None:
        logger.debug('Downloading crypto historical data')
        try:
            if config:
                self.start = config.start
                self.end = config.end
                self.source = config.source
                self.ticker = config.ticker
                self.category = config.category
                self.savePath = config.savePath
            else:
                raise Exception(f'Error in the input configuration: {config!r}')
        except Exception as e:
            logger.critical(e)

        try:
            self.download()
        except Exception as e:
            logger.error(e)

    def download():
        pass


class ForexHistoricalData():
    def __init__(self, config=None) -> None:
        logger.debug('Downloading forex historical data')

        try:
            if config:
                self.start      = config['start']
                self.end        = config['end']
                self.source     = config['source']
                self.ticker     = config['ticker']
                self.fields     = config['fields']
                self.vendorFields     = config['vendorFields']
                self.freq       = config['freq']
                self.category   = config['category']
                self.savePath   = config['savePath']
            else:
                raise Exception(f'Error in the input configuration: {config!r}')
        except Exception as e:
            logger.critical(e)

        ForexHistoricalData.download(self)
        # try:
        #     ForexHistoricalData.download(self)
        # except Exception as e:
        #     logger.error(e)

    def download(self):
        try:
            md_request = MarketDataRequest(start_date=self.start, finish_date=self.end,
                                        fields=self.fields, vendor_fields=self.vendorFields,
                                        freq=self.freq, data_source=self.source,
                                        tickers=self.ticker, vendor_tickers=self.ticker)
            logger.debug('Fetching data')
            df = market.fetch_market(md_request)
            logger.debug(f'Data shape: {df.shape}')
        except Exception as e:
            logger.error(f'Error in downloading data, error: {e}')
        
        df.columns = self.fields
        df = df.rename_axis('date')
        print(df.head())

        if os.path.isfile(self.savePath):
            with open(self.savePath, 'rb') as file:
                olddf = pickle.load(file)
                df.append(olddf)
                df.sort_values(by='date')
                df.drop_duplicates()
        
        df.head()
        df.tail()
        with open(self.savePath, 'wb') as file:
            pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f'Saved to file: {self.savePath}')



if __name__ == "__main__":
    saveDir = "D:\MarketData"

    """CRYPTO"""
    # start = ''
    # end = ''
    # ticker = 'ETHUSD'
    # fileName = 'CRYPTO'

    # config = {'start': '',
    #           'end': '',
    #           'freq': '',
    #           'source': '',
    #           'ticker': '',
    #           'category': '',
    #           'savePath': os.path.join(saveDir, fileName)}
    # CryptoHistoricalData(config)

    """FOREX"""
    saveDir = "D:\MarketData\FX"
    start = '1 Jan 2010'
    start = '1 Jan 2019'
    end = '15 Jul 2021'
    #tickers = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
    tickers = ['GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
    freq = 'tick'

    for ticker in tickers:
        fileName = f'{ticker}_{freq}.pkl'
        savePath = os.path.join(saveDir, fileName)

        config = {'start': start,
                'end': end,
                'freq': freq,
                'fields': ['bid', 'ask', 'volume'],
                'vendorFields': ['bid', 'ask', 'settle'],
                'source': 'dukascopy',
                'ticker': ticker,
                'category': 'forex',
                'savePath': savePath}
        ForexHistoricalData(config)
