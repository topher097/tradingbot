from BinanceConnect import *
<<<<<<< HEAD
from KlineData import *
from Technical import *
import json
import pickle
import os
import csv
from datetime import datetime
from matplotlib import pyplot as plt
=======
>>>>>>> parent of 5476c99 (added pickle historical data)


class TradingBot():
    def __init__(self):
<<<<<<< HEAD
        self.credentialsFileName    = "bin/credentials.json"
        self.methodFileName         = "bin/methods.json"
        self.tradePairsFileName     = "bin/tradingpairs.json"
        self.availPairsFileName     = "bin/availabletradingpairs.json"
        self.BASEDIR                = os.path.dirname(os.path.realpath(__file__))
        self.tradingPairs           = []
        self.tradingCoins           = []
        self.kLinesData             = {}
        self.kLinesTA               = {}
        self.methods                = {}

        
        """ Run initialization methods """
        BinanceConnect.createClient(self, self.credentialsFileName)     # Connect to binance via API
        TradingBot.getTradingPairsAndCoins(self)                        # Get the target trading pairs to trade
        TradingBot.loadMethodsJSON(self)                                # Load the current methods
        TradingBot.backtestMethods(self, pairs=['ETHUSDC'], methods=self.methods)
        #KlineData.saveHistoricalData(self, self.tradingPairs)
        #TradingBot.test(self)
        #TradingBot.backtestMethods(self, pairs=self.tradingPairs)

    def loadMethodsJSON(self):
        # Load the methods and their values from JSON file
        try:
            with open(self.methodFileName, "r") as file:
                data            = file.read()
                self.methods    = json.loads(data)
            print(f"Successfully loaded methods from {self.methodFileName}")
        except Exception as e:
            print(f"Failed to load methods, error: {e}")
    
    def saveMethodsJSON(self):
        # Save the current methods to the JSON file
        try:
            with open(self.methodFileName, "w") as file:
                json.dump(self.methods, file)
            print(f"Successfully saved methods to {self.methodFileName}")
        except Exception as e:
            print(f"Failed to save methods, error: {e}")
    
=======
        self.tradingPairs
        
>>>>>>> parent of 5476c99 (added pickle historical data)
        
        BinanceConnect.createClient(self)           # Connect to binance via api
        self.getTradingPairs(self)                  # Get the valid trading pairs to trade
              
        
    def getTradingPairs(self):
        # Get list of target trading pairs
        with open(self.tradePairsFileName, "r") as file:
            data                = file.read()
            self.tradingPairs   = json.loads(data)["pairs"]
<<<<<<< HEAD
            self.tradingCoins   = json.loads(data)["coins"]
            print(f"Loaded target trading pairs: {', '.join(self.tradingPairs)}")
            print(f"Loaded coins: {', '.join(self.tradingCoins)}")
    
    def backtestMethods(self, pairs, methods=['RSI']):
        # Get historical data for trading pairs given parameters below:
        timeInterval    = '1h'
        start           = 'Jan 1, 2021'
        stop            = 'Jun 1, 2021'
        for pair in pairs:
            print(f"Backtesting methods on {pair}")
            kLines = KlineData.loadHistoricalCSVData(self, pair, timeInterval, start, stop)
            if not kLines.any():
                print(f"kLine data empty, going to next pair...")
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
        print(f"Finished loading kLine data for pairs: [{', '.join(pairs)}] given paramaters time interval: {timeInterval}, start: {start}, and stop: {stop}")
    
    def plotKlineData(self, kLines):
        pass


    def test(self):
        print(self.methods)

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
                


        
        
=======

    def test(self):
        BinanceConnect.getAllPrices(self)
        BinanceConnect.get
        print(self.tradingPairs)
        print()
>>>>>>> parent of 5476c99 (added pickle historical data)


if "__name__" == "__main__":
    bot = TradingBot()