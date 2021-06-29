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
                self.client = Client(api_key, api_secret, tld='us', requests_params={"verify": True, "timeout": 20})
        except Exception as e:
            logger.critical(e)
            exit()
        else:
            logger.info("Successful client creation")

    def getSystemStatus(self):
        try:
            result = self.client.get_system_status()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getServerTime(self):
        try:
            result = self.client.get_server_time()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
        
    def pingServer(self):
        try:
            self.client.ping()
            logger.debug('pinged server')
        except Exception as e:
            logger.error(e)

    def getExchangeInfo(self):
        try:
            result = self.client.get_exchange_info()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    """ *************** ACCOUNT METHODS *************** """
    def getAccountInfo(self):
        try:
            result = self.client.get_account()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getAccountStatus(self):
        try:
            result = self.client.get_account_status()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getAccountTradingStatus(self):
        try:
            result = self.client.get_account_api_trading_status()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
    
    def getAssetDetails(self):
        try:
            result = self.client.get_asset_details()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getAssetBalance(self, asset='USD'):
        try:
            result = self.client.get_asset_balance(asset=asset)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e) 

    def getCoinAddress(self, coin):
        try:
            result = self.client.get_deposit_address(coi=coin)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getWithdrawHistory(self, coin):
        try:
            result = self.client.get_withdraw_history(coin=coin)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    """ *************** MARKET METHODS *************** """
    def getDepth(self, pair):
        try:
            result = self.client.get_order_book(symbol=pair)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getPrice(self, pair):
        try:
            result = self.client.get_ticker(symbol=pair)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
    
    def getAllPrices(self):
        try:
            result = self.client.get_all_tickers()
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    """ *************** TRADING & ORDER METHODS *************** """
    def getTradesOfPair(self, pair):
        try:
            result = self.client.get_my_trades(symbol=pair)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getTradeFeesOfPair(self, pair):
        try:
            result = self.client.get_trade_fee(symbol=pair)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def fetchAllPairOrders(self, pair, limit=10):
        try:
            result = self.client.get_all_orders(symbol=pair, limit=limit)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def placeOrder(self, pair=None, side=None, orderType=None, timeInForce=TIME_IN_FORCE_GTC, quantity=None, price=None):
        try:
            self.client.create_order(symbol=pair,
                                     side=side,
                                     type=orderType,
                                     timeInForce=timeInForce,
                                     quantity=quantity,
                                     price=price)
            logger.info(f"Order created: {pair}, {side}, {orderType}, {timeInForce}, {quantity}, {price}")
            
        except Exception as e:
            print(e)
            logger.error(e)

    def placeTestOrder(self, pair=None, side=None, orderType=None, timeInForce=TIME_IN_FORCE_GTC, quantity=None, price=None):
        try:
            self.client.create_test_order(symbol=pair,
                                          side=side,
                                          type=orderType,
                                          timeInForce=timeInForce,
                                          quantity=quantity,
                                          price=price)
            logger.info(f"Test order created: {pair}, {side}, {orderType}, {timeInForce}, {quantity}, {price}")
        except Exception as e:
            logger.error(e)

    def cancelOrder(self, pair, orderID):
        try:
            self.client.cancel_order(symbol=pair, orderID=orderID)
            logger.info(f"Order cancelled: {pair}, {orderID}")
        except Exception as e:
            logger.error(e)

    def getOrderStatus(self, pair, orderID):
        try:
            result = self.client.get_order(symbol=pair, orderID=orderID)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
    
    def getOpenPairOrders(self, pair):
        try:
            result = self.client.get_open_orders(symbol=pair)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getAllPairOrders(self, pair):
        try:
            result = self.client.get_all_orders(symbol=pair)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
        







