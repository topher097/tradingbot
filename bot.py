from BinanceConnect import *

class TradingBot():
    def __init__(self):
        self.tradingPairs
        
        
        BinanceConnect.createClient(self)           # Connect to binance via api
        self.getTradingPairs(self)                  # Get the valid trading pairs to trade
              
        
    def getTradingPairs(self):
        # Get list of target trading pairs
        with open("tradingpairs.json", "r") as file:
            data                = file.read()
            self.tradingPairs   = json.loads(data)["pairs"]

    def test(self):
        BinanceConnect.getAllPrices(self)
        BinanceConnect.get
        print(self.tradingPairs)
        print()


if "__name__" == "__main__":
    bot = TradingBot()