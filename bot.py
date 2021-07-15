from BinanceConnect import *
from KlineData import *
from Technical import *
import Training
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import json
import os
from loggerSettings import logger
from datetime import datetime
import matplotlib.pyplot as plt


class TradingBot():
    def __init__(self):
        self.credentialsFileName    = "bin/credentials.json"
        self.methodFileName         = "bin/methods.json"
        self.tradePairsFileName     = "bin/tradingpairs.json"
        self.availPairsFileName     = "bin/availabletradingpairs.json"
        self.accountReportFileName  = "bin/accountReport.json"
        self.histCSVDir             = "historical_data_csv"
        self.histPklDir             = "historical_data"
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
        TradingBot.backtestMethods(self, pairs=self.tradingPairs, methods=self.methods)
        #TradingBot.backtestMethods(self, pairs=['ETHUSDC'], methods=self.methods)
        #KlineData.saveHistoricalData(self, self.tradingPairs)
        #TradingBot.test(self)
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
        start           = 'Jan 1, 2017'
        stop            = 'Jun 1, 2021'
        # Iterate through pairs data
        for pair in pairs:
            logger.info(f"Backtesting methods on {pair}")
            kLines = KlineData.loadHistoricalCSVData(self, pair, timeInterval, start, stop)
            if not kLines.any():
                logger.warning(f"kLine data empty, going to next pair...")
                break
            else:
                logger.info(f"Finished loading kLine data for pair: {pair} given paramaters time interval: {timeInterval}, start: {start}, and stop: {stop}")
            # Create model file name
            methodStrings = ""
            if 'RSI' in methods: methodStrings += f"_RSI_{methods['RSI']['type']}_{methods['RSI']['timePeriod']}"
            if 'PSAR' in methods: methodStrings += f"_PSAR_{methods['PSAR']['acceleration']}_{methods['PSAR']['maximum']}"

            #modelFileName = f"models\BiLSTM\{pair}\{pair}_{timeInterval}_{datetime.strptime(start, '%b %d, %Y').strftime('%m%d%Y')}_{datetime.strptime(stop, '%b %d, %Y').strftime('%m%d%Y')}{methodStrings}.hdf5"
            modelFileName = f"models\BiLSTM\{timeInterval}_{datetime.strptime(start, '%b %d, %Y').strftime('%m%d%Y')}_{datetime.strptime(stop, '%b %d, %Y').strftime('%m%d%Y')}{methodStrings}.hdf5"
            modelFilePath = os.path.join(self.BASEDIR, modelFileName)
            # Run or get model from file
            try:
                df_train, train_data, df_val, val_data, df_test, test_data = Training.TrainingMisc.preprocessData(self, klines=kLines, pair=pair, methods=methods, plot=False)
                Training.BiDirectionalLSTM(train_data, val_data, test_data, modelFilePath, pair, plotEval=True)
            except Exception as e:
                logger.error(f"Error while training model, error: {e}")
        
        # Show all plots
        plt.show()            

        
    
    def plotKlineData(self, kLines):
        pass

    def test(self):
        #BinanceConnect.getSystemStatus(self)       # error
        #BinanceConnect.getAssetDetails(self)       # error
        #BinanceConnect.getAccountTradingStatus(self)    # error
        lot = BinanceConnect.calcMinLotParamsAtMarketPrice(self, pair='VETUSD', minLotPrice=10.1)
        BinanceConnect.placeTestOrder(self, 
                                      pair=lot['symbol'], 
                                      side=Client.SIDE_BUY, 
                                      orderType=Client.ORDER_TYPE_MARKET, 
                                      timeInForce=TIME_IN_FORCE_GTC, 
                                      quantity=lot['qty'], 
                                      price=lot['price'])
        
    def deleteHistoricalData(self):
        # Delete pickle files
        dirPath = os.path.join(self.BASEDIR,self.histCSVDir)
        keep = "01012017"
        print(dirPath)
        print(os.listdir(dirPath))
        for file in os.listdir(dirPath):
            if keep not in file:
                filePath = os.path.join(dirPath, file)
                try:
                    if os.path.isfile(filePath):
                        os.remove(filePath)
                    else:
                        raise Exception("file doesn't exist")
                except Exception as e:
                    print(f"cannot delete {filePath}, reason: {e}")

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