import asyncio
import json
from decimal import Decimal as D, ROUND_DOWN, ROUND_UP
import decimal
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
                return self.client
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

    def getPairPrice(self, pair):
        try:
            result = self.client.get_symbol_ticker(symbol=pair)['price']
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

    def getPairInfo(self, pair):
        try:
            result = self.client.get_symbol_info(symbol=pair)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
    
    def getPairMinQty(self, pair):
        try:
            result = self.client.get_symbol_info(symbol=pair)['filters'][2]['minQty']
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
    
    def getPairMaxQty(self, pair):
        try:
            result = self.client.get_symbol_info(symbol=pair)['filters'][2]['maxQty']
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def getPriceChange(self, pair):
        try:
            result = self.client.get_ticker(symbol=pair)
            self.client.get_sy
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

    def placeOrder(self, pair=None, side=None, orderType=None, timeInForce=None, quantity=None, price=None):
        try:
            if orderType != Client.ORDER_TYPE_MARKET:
                self.client.create_order(symbol=pair,
                                        side=side,
                                        type=orderType,
                                        timeInForce=timeInForce,
                                        quantity=quantity,
                                        price=price)
                logger.info(f"Order created: {pair}, {side}, {orderType}, {timeInForce}, {quantity}, {price} with USD val of ${round(float(price)*float(quantity),2)}")
            else:
                self.client.create_order(symbol=pair,
                                        side=side,
                                        type=orderType,
                                        quantity=quantity)
                price = self.client.get_symbol_ticker(symbol=pair)['price']
                logger.info(f"Order created: {pair}, {side}, {orderType}, {quantity}, {price} with USD val of ${round(float(price)*float(quantity),2)}")
        except Exception as e:
            print(e)
            logger.error(e)

    def placeTestOrder(self, pair=None, side=None, orderType=None, timeInForce=None, quantity=None, price=None):
        try:
            if orderType != Client.ORDER_TYPE_MARKET:
                self.client.create_test_order(symbol=pair,
                                            side=side,
                                            type=orderType,
                                            timeInForce=timeInForce,
                                            quantity=quantity,
                                            price=price)
                logger.info(f"Test order created: {pair}, {side}, {orderType}, {timeInForce}, {quantity}, {price} with USD val of ${round(float(price)*float(quantity),2)}")
            else:
                self.client.create_test_order(symbol=pair,
                                            side=side,
                                            type=orderType,
                                            quantity=quantity)
                price = self.client.get_symbol_ticker(symbol=pair)['price']
                logger.info(f"Test order created: {pair}, {side}, {orderType}, {quantity}, {price} with USD val of ${round(float(price)*float(quantity),2)}")
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

    def calcPairMinQty(self, pair, minLotPrice=10.30):
        try:
            enforcedMinQty = float(self.client.get_symbol_info(symbol=pair)['filters'][2]['minQty'])
            currentPrice   = float(self.client.get_symbol_ticker(symbol=pair)['price'])
            minQty = minLotPrice/currentPrice
            if minQty < enforcedMinQty: minQty = enforcedMinQty
            result = str(minQty)
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)

    def calcMinLotParamsAtMarketPrice(self, pair, minLotPrice=10.30):
        try:
            pairInfo = self.client.get_symbol_info(symbol=pair)
            lotPrecision   = float(pairInfo['filters'][2]['stepSize'])
            pricePrecision    = float(pairInfo['filters'][0]['tickSize'])
            enforcedMinQty = float(pairInfo['filters'][2]['minQty'])
            currentPrice   = D.from_float(float(self.client.get_symbol_ticker(symbol=pair)['price'])).quantize(D(str(pricePrecision)))
            minQty = float(minLotPrice)/float(currentPrice)
            if minQty < float(enforcedMinQty): minQty = float(enforcedMinQty)   # If minQty less than enforced min qty, set equal to enforced min qty
            if int(lotPrecision) >= 1: qty = int(minQty)                        # If lot precision bigger than 1.0, get int of minQty
            else: qty = D.from_float(minQty).quantize(D(str(lotPrecision)))     # if lot precision less than 1.0, get correct precision of minQty
            result = {'symbol':pair, 'price':str(currentPrice), 'qty':str(qty)}
            logger.debug(result)
            return result
        except Exception as e:
            logger.error(e)
        







