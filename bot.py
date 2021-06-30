from BinanceConnect import *
from KlineData import *
from Technical import *
import json
import os
from loggerSettings import logger
from datetime import datetime
from matplotlib import pyplot as plt


class TradingBot():
    def __init__(self):
        self.credentialsFileName    = "bin/credentials.json"
        self.methodFileName         = "bin/methods.json"
        self.tradePairsFileName     = "bin/tradingpairs.json"
        self.availPairsFileName     = "bin/availabletradingpairs.json"
        self.accountReportFileName  = "bin/accountReport.json"
        self.BASEDIR                = os.path.dirname(os.path.realpath(__file__))
        self.tradingPairs           = []
        self.tradingAssets          = []
        self.kLinesData             = {}
        self.kLinesTA               = {}
        self.methods                = {}

        """ Initiate Logging File """
        logger.info("starting bot")
        
        """ Run initialization methods """
        self.client = BinanceConnect.createClient(self, self.credentialsFileName)     # Connect to binance via API
        TradingBot.getTradingPairsAndAssets(self)                                     # Get the target trading pairs to trade
        TradingBot.loadMethodsJSON(self)                                              # Load the current methods
        #TradingBot.backtestMethods(self, pairs=['ETHUSDC'], methods=self.methods)
        #KlineData.saveHistoricalData(self, self.tradingPairs)
        TradingBot.test(self)
        #TradingBot.backtestMethods(self, pairs=self.tradingPairs)

    def loadMethodsJSON(self):
        # Load the methods and their values from JSON file
        try:
            with open(self.methodFileName, "r") as file:
                data            = file.read()
                self.methods    = json.loads(data)
            logger.info(f"Successfully loaded methods from {self.methodFileName}")
        except Exception as e:
            logger.error(f"Failed to load methods, error: {e}")
    
    def saveMethodsJSON(self):
        # Save the current methods to the JSON file
        try:
            with open(self.methodFileName, "w") as file:
                json.dump(self.methods, file)
            logger.info(f"Successfully saved methods to {self.methodFileName}")
        except Exception as e:
            logger.warning(f"Failed to save methods, error: {e}")             
        
    def getTradingPairsAndAssets(self):
        # Get list of target trading pairs
        with open(self.tradePairsFileName, "r") as file:
            data                = file.read()
            self.tradingPairs   = json.loads(data)["pairs"]
            self.tradingCoins   = json.loads(data)["assets"]
            logger.info(f"Loaded target trading pairs: {', '.join(self.tradingPairs)}")
            logger.info(f"Loaded assets: {', '.join(self.tradingAssets)}")
    
    def backtestMethods(self, pairs, methods=['RSI']):
        # Get historical data for trading pairs given parameters below:
        timeInterval    = '1h'
        start           = 'Jan 1, 2021'
        stop            = 'Jun 1, 2021'
        # Iterate through pairs data
        for pair in pairs:
            logger.info(f"Backtesting methods on {pair}")
            kLines = KlineData.loadHistoricalCSVData(self, pair, timeInterval, start, stop)
            if not kLines.any():
                logger.warning(f"kLine data empty, going to next pair...")
                break
            self.kLinesData[pair] = kLines
            self.kLinesTA[pair] = {}
            # Get RSI of historical data
            if 'RSI' in methods:     self.kLinesTA[pair]['RSI'] = Technical.getRSI(self, kLines, type=self.methods['RSI']['type'], timePeriod=self.methods['RSI']['timePeriod'])
            else:                    self.kLinesTA[pair]['RSI'] = []
            # Get Parabolic SAR of historical data
            if 'PSAR' in methods:    self.kLinesTA[pair]['PSAR'] = Technical.getParabolicSAR(self, kLines, acceleration=self.methods['PSAR']['acceleration'], maximum=self.methods['PSAR']['maximum'])
            else:                    self.kLinesTA[pair]['PSAR'] = []
            for key in methods:
                print(key, self.kLinesTA[pair][key])
            

        #plt.show()
        logger.info(f"Finished loading kLine data for pairs: [{', '.join(pairs)}] given paramaters time interval: {timeInterval}, start: {start}, and stop: {stop}")
    
    def plotKlineData(self, kLines):
        pass

    def test(self):
        #BinanceConnect.pingServer(self)
        #BinanceConnect.getAccountInfo(self)
        #BinanceConnect.getServerTime(self)
        #BinanceConnect.getSystemStatus(self)       # error
        #BinanceConnect.getAssetDetails(self)       # error
        #BinanceConnect.getWithdrawHistory(self, coin='ADA')
        #BinanceConnect.getAssetBalance(self, asset='VET')
        #BinanceConnect.getAccountTradingStatus(self)    # error
        lot = BinanceConnect.calcMinLotParamsAtMarketPrice(self, pair='VETUSD', minLotPrice=10.1)
        BinanceConnect.placeTestOrder(self, 
                                      pair=lot['symbol'], 
                                      side=Client.SIDE_BUY, 
                                      orderType=Client.ORDER_TYPE_MARKET, 
                                      timeInForce=TIME_IN_FORCE_GTC, 
                                      quantity=lot['qty'], 
                                      price=lot['price'])
        

    def updateAccountReport(self):
        with open(self.accountReportFileName, 'r+') as file:
            fileData= json.load(file)
            time = int(datetime.strftime("%s"))
            body = 1
            newEntry = {str(datetime.strftime("%s")) : {"USD Bal" : BinanceConnect.getAssetBalance(asset='USD')}}
            fileData.update(newEntry)
            file.seek(0)
            json.dump(fileData, file)


    def createLegalPairsJSON(self):
        prices = BinanceConnect.getAllPrices(self)
        pairs = []
        for data in prices:
            pairs.append(data["symbol"])
        filedata = {}
        filedata["pairs"] = pairs
        jstr = json.dumps(filedata, indent=4)
        with open(self.availPairsFileName, "w") as file:
            file.write(jstr)
                


if __name__ == "__main__":
    bot = TradingBot()