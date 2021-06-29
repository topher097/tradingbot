import asyncio
import json
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance import AsyncClient, DepthCacheManager, BinanceSocketManager
from binance.enums import *
from loggerSettings import logger


class BinanceConnect():
    def __init__(self):
        self.client

    """ *************** CLIENT METHODS *************** """
    def createClient(self, credentialsFileName):
        # Create client session using api credentials
        try:
            # Insert api_key and api_secret into file called credentials.json
            # See bin/example_credentials.json for reference
            with open(credentialsFileName, "r") as file:
                data        = file.read()
                creds       = json.loads(data)
                api_key     = creds["api_key"]
                api_secret  = creds["api_secret"]
                self.client = Client(api_key, api_secret)
        except Exception as e:
            logger.critical(e)
            exit()
        else:
            logger.info("Successful client creation")

    def getSystemStatus(self):
        return self.client.get_system_status()

    def getServerTime(self):
        try:
            result = self.client.get_server_time()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
        
    
    def pingServer(self):
        self.client.ping()

    def getExchangeInfo(self):
        return self.client.get_exchange_info()

    """ *************** ACCOUNT METHODS *************** """
    def getAccountInfo(self):
        return self.client.get_account()

    def getAccountStatus(self):
        return self.client.get_account_status()

    def getAccountTradingStatus(self):
        return self.client.get_account_api_trading_status()
    
    def getAssetDetails(self):
        return self.client.get_asset_details()

    def getAssetBalance(self, asset='USD'):
        return self.client.get_asset_balance(asset=asset) 

    def getCoinAddress(self, coin):
        return self.client.get_deposit_address(coi=coin)

    def getWithdrawHistory(self, coin):
        return self.client.get_withdraw_history(coin=coin)

    """ *************** MARKET METHODS *************** """
    def getDepth(self, pair):
        return self.client.get_order_book(symbol=pair)

    def getPrice(self, pair):
        return self.client.get_ticker(symbol=pair)
    
    def getAllPrices(self):
        return self.client.get_all_tickers()

    """ *************** TRADING & ORDER METHODS *************** """
    def getTradesOfPair(self, pair):
        return self.client.get_my_trades(symbol=pair)

    def getTradeFeesOfPair(self, pair):
        return self.client.get_trade_fee(symbol=pair)

    def fetchAllPairOrders(self, pair, limit=10):
        return self.client.get_all_orders(symbol=pair, limit=limit)

    def placeOrder(self, pair=None, side=None, orderType=None, timeInForce=TIME_IN_FORCE_GTC, quantity=None, price=None):
        try:
            self.client.create_order(symbol=pair,
                                     side=side,
                                     type=orderType,
                                     timeInForce=timeInForce,
                                     quantity=quantity,
                                     price=price)
            return f"Order created: {pair}, {side}, {orderType}, {timeInForce}, {quantity}, {price}"
        except Exception as e:
            return e

    def cancelOrder(self, pair, orderID):
        try:
            self.client.cancel_order(symbol=pair, orderID=orderID)
            return f"Order cancelled: {pair}, {orderID}"
        except Exception as e:
            return e

    def getOrderStatus(self, pair, orderID):
        return self.client.get_order(symbol=pair, orderID=orderID)
    
    def getOpenPairOrders(self, pair):
        return self.client.get_open_orders(symbol=pair)

    def getAllPairOrders(self, pair):
        return self.client.get_all_orders(symbol=pair)
        







