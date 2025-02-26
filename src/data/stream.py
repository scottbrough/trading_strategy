"""
Real-time data streaming module for handling live market data.
Uses WebSocket connections to Kraken's API with automatic reconnection and error handling.
"""

import websockets
import asyncio
import json
import hmac
import hashlib
import base64
import time
from typing import Dict, List, Callable, Any
from datetime import datetime
import pandas as pd
from collections import defaultdict
import threading
from queue import Queue

from ..core.config import config
from ..core.logger import log_manager
from ..core.exceptions import DataError
from .database import db

logger = log_manager.get_logger(__name__)

class KrakenStreamManager:
    """Data streaming manager for Kraken exchange"""
    async def get_exchange_info(self):
        """Get exchange trading rules and info"""
        try:
            await self.ensure_connection()
            
            info_message = {
                "event": "info",
                "reqid": int(time.time())
            }
            await self.ws.send(json.dumps(info_message))
            
            response = await self.ws.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to get exchange info: {str(e)}")
            return None

    async def get_account_balance(self):
        """Get account balance information"""
        try:
            await self.ensure_connection()
            
            # Generate authentication
            nonce = str(int(time.time() * 1000))
            message = nonce + "balance"
            signature = self._generate_signature(message)
            
            balance_message = {
                "event": "balance",
                "token": self.api_key,
                "nonce": nonce,
                "signature": signature
            }
            
            await self.ws.send(json.dumps(balance_message))
            response = await self.ws.recv()
            
            # Parse and return balance data
            balance_data = json.loads(response)
            if "errorMessage" in balance_data:
                logger.error(f"Balance request error: {balance_data['errorMessage']}")
                return None
            
            return balance_data.get("balances", {})
        except Exception as e:
            logger.error(f"Failed to get account balance: {str(e)}")
            return None
    async def market_buy(self, symbol, size):
        """Place market buy order"""
        order_data = {
            "symbol": symbol,
            "action": "buy",
            "type": "market",
            "size": size
        }
        return await self.place_order(order_data)

    async def market_sell(self, symbol, size):
        """Place market sell order"""
        order_data = {
            "symbol": symbol,
            "action": "sell",
            "type": "market",
            "size": size
        }
        return await self.place_order(order_data)

    async def limit_buy(self, symbol, size, price, time_in_force="GTC"):
        """Place limit buy order"""
        order_data = {
            "symbol": symbol,
            "action": "buy",
            "type": "limit",
            "size": size,
            "price": price,
            "time_in_force": time_in_force
        }
        return await self.place_order(order_data)

    async def limit_sell(self, symbol, size, price, time_in_force="GTC"):
        """Place limit sell order"""
        order_data = {
            "symbol": symbol,
            "action": "sell",
            "type": "limit",
            "size": size,
            "price": price,
            "time_in_force": time_in_force
        }
        return await self.place_order(order_data)

    class OrderManager:
        def __init__(self):
            self.open_orders = {}
            self.order_history = []
            self.lock = asyncio.Lock()
        
        async def add_order(self, order):
            """Add a new order to tracking"""
            async with self.lock:
                order_id = order.get("order_id")
                if order_id:
                    self.open_orders[order_id] = order
                    return order_id
        
        async def update_order(self, order_id, status, fill_price=None):
            """Update order status"""
            async with self.lock:
                if order_id in self.open_orders:
                    self.open_orders[order_id]["status"] = status
                    if fill_price:
                        self.open_orders[order_id]["fill_price"] = fill_price
                    if status in ["filled", "canceled", "rejected"]:
                        order = self.open_orders.pop(order_id)
                        order["update_time"] = datetime.now()
                        self.order_history.append(order)
                    return True
                return False
        
        def get_open_orders(self):
            """Get all open orders"""
            return list(self.open_orders.values())
        
        def get_order_history(self, limit=100):
            """Get order history"""
            return self.order_history[-limit:]

    async def ensure_connection(self):
        """Ensure WebSocket connection is active and stable"""
        if self.ws is None or not self.running:
            logger.info("Establishing new WebSocket connection")
            return await self.connect()
        
        try:
            # Ping to verify connection
            await self.ws.ping()
            return True
        except Exception as e:
            logger.warning(f"Connection check failed: {str(e)}")
            self.ws = None
            return await self.connect()

    async def place_order(self, order_data):
        """Place an order via WebSocket API"""
        try:
            await self.ensure_connection()
            
            order_message = {
                "event": "addOrder",
                "token": self.api_key,
                "ordertype": order_data.get("type", "limit"),
                "pair": order_data["symbol"],
                "price": str(order_data.get("price", 0)),
                "volume": str(order_data["size"]),
                "side": order_data["action"],
                "leverage": order_data.get("leverage", "none"),
                "timeinforce": order_data.get("time_in_force", "GTC")
            }
            
            # Sign the message
            nonce = str(int(time.time() * 1000))
            message = nonce + json.dumps(order_message)
            order_message["nonce"] = nonce
            order_message["signature"] = self._generate_signature(message)
            
            # Send the order
            await self.ws.send(json.dumps(order_message))
            
            # Wait for response
            response = await self.ws.recv()
            return self._parse_order_response(response)
            
        except Exception as e:
            logger.error(f"Order placement failed: {str(e)}")
            raise DataError.DataStreamError("Failed to place order", details={"error": str(e)})

    def _parse_order_response(self, response):
        """Parse order response from Kraken"""
        try:
            data = json.loads(response)
            if data.get("event") == "addOrderStatus" and data.get("status") == "ok":
                return {
                    "order_id": data.get("txid"),
                    "status": "placed",
                    "timestamp": datetime.now()
                }
            else:
                logger.error(f"Order placement error: {data.get('errorMessage', 'Unknown error')}")
                return {
                    "status": "failed",
                    "error": data.get("errorMessage", "Unknown error"),
                    "timestamp": datetime.now()
                }
        except Exception as e:
            logger.error(f"Error parsing order response: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def __init__(self):
        self.config = config
        self.api_key = config.get('exchange.api_key')
        self.api_secret = config.get('exchange.api_secret')
        self.ws_url = config.get('exchange.websocket_url')
        self.sandbox = config.is_sandbox()
        
        # Data storage
        self.latest_data = defaultdict(dict)
        self.orderbook = defaultdict(dict)
        self.trades = defaultdict(list)
        
        # Callbacks
        self.callbacks = defaultdict(list)
        
        # Connection management
        self.ws = None
        self.running = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 300  # 5 minutes
        
        # Message queue for processing
        self.message_queue = Queue()
        self.processing_thread = None

        self.order_manager = self.OrderManager()
        # Add to KrakenStreamManager __init__
        self.health_monitor = self.ConnectionHealthMonitor()
    
    async def connect(self):
        """Establish WebSocket connection with authentication"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            
            if self.sandbox:
                logger.info("Connected to Kraken sandbox WebSocket")
            else:
                # Production authentication
                timestamp = int(time.time() * 1000)
                nonce = str(timestamp)
                
                # Create signature
                signature = self._generate_signature(nonce)
                
                # Send authentication message
                auth_message = {
                    "event": "subscribe",
                    "apiKey": self.api_key,
                    "nonce": nonce,
                    "signature": signature
                }
                await self.ws.send(json.dumps(auth_message))
                response = await self.ws.recv()
                
                if not self._verify_auth(response):
                    raise DataError.DataStreamError(
                        "Authentication failed",
                        details={"response": response}
                    )
                
                logger.info("Successfully authenticated with Kraken WebSocket")
            
            self.running = True
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            return False
    
    def _generate_signature(self, nonce: str) -> str:
        """Generate signature for authentication"""
        try:
            message = nonce.encode('utf-8')
            secret = base64.b64decode(self.api_secret)
            signature = hmac.new(secret, message, hashlib.sha256).digest()
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate signature: {str(e)}")
            raise DataError.DataStreamError("Signature generation failed")
    
    def _verify_auth(self, response: str) -> bool:
        """Verify authentication response"""
        try:
            resp_data = json.loads(response)
            return resp_data.get('event') == 'subscriptionStatus' and resp_data.get('status') == 'subscribed'
        except json.JSONDecodeError:
            return False
    
    async def subscribe(self, symbols: List[str], channels: List[str]):
        """Subscribe to specified channels for given symbols"""
        try:
            for symbol in symbols:
                for channel in channels:
                    subscription = {
                        "event": "subscribe",
                        "pair": [symbol],
                        "subscription": {"name": channel}
                    }
                    await self.ws.send(json.dumps(subscription))
                    logger.info(f"Subscribed to {channel} for {symbol}")
                    
                    # Wait for subscription confirmation
                    response = await self.ws.recv()
                    if not self._verify_subscription(response, symbol, channel):
                        logger.warning(f"Subscription failed for {symbol} {channel}")
            
        except Exception as e:
            logger.error(f"Subscription failed: {str(e)}")
            raise DataError.DataStreamError("Failed to subscribe to channels")
    
    def _verify_subscription(self, response: str, symbol: str, channel: str) -> bool:
        """Verify subscription response"""
        try:
            resp_data = json.loads(response)
            return (resp_data.get('event') == 'subscriptionStatus' and
                   resp_data.get('pair') == symbol and
                   resp_data.get('status') == 'subscribed')
        except json.JSONDecodeError:
            return False
    
    async def _handle_message(self, message: str):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if isinstance(data, list):
                # Trading data
                channel_name = data[2]
                symbol = data[3]
                
                if channel_name == "trade":
                    self._process_trade(symbol, data[1])
                elif channel_name == "ohlc":
                    self._process_ohlc(symbol, data[1])
                elif channel_name == "book":
                    self._process_orderbook(symbol, data[1])
                
                # Add to message queue for processing
                self.health_monitor.update()
                
            elif isinstance(data, dict):
                # System messages
                if data.get('event') == 'systemStatus':
                    logger.info(f"System status: {data.get('status')}")
                elif data.get('event') == 'error':
                    logger.error(f"WebSocket error: {data.get('errorMessage')}")
        
        except json.JSONDecodeError:
            logger.error("Failed to decode WebSocket message")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _process_trade(self, symbol: str, trade_data: List):
        """Process trade data"""
        try:
            for trade in trade_data:
                trade_record = {
                    'price': float(trade[0]),
                    'volume': float(trade[1]),
                    'time': datetime.fromtimestamp(float(trade[2])),
                    'side': 'buy' if trade[3] == 'b' else 'sell',
                    'type': 'market' if trade[4] == 'm' else 'limit'
                }
                self.trades[symbol].append(trade_record)
                
                # Store in database
                db.store_trade({
                    'symbol': symbol,
                    'side': trade_record['side'],
                    'price': trade_record['price'],
                    'amount': trade_record['volume'],
                    'timestamp': trade_record['time'],
                    'type': trade_record['type']
                })
                
                # Trigger callbacks
                for callback in self.callbacks['trade']:
                    callback(symbol, trade_record)
                    
        except Exception as e:
            logger.error(f"Error processing trade: {str(e)}")
    
    def _process_ohlc(self, symbol: str, ohlc_data: List):
        """Process OHLCV data"""
        try:
            candle = {
                'time': datetime.fromtimestamp(float(ohlc_data[0])),
                'open': float(ohlc_data[1]),
                'high': float(ohlc_data[2]),
                'low': float(ohlc_data[3]),
                'close': float(ohlc_data[4]),
                'volume': float(ohlc_data[6])
            }
            self.latest_data[symbol].update(candle)
            
            # Store in database
            df = pd.DataFrame([candle], index=[candle['time']])
            db.store_ohlcv(symbol, '1m', df)
            
            # Trigger callbacks
            for callback in self.callbacks['ohlc']:
                callback(symbol, candle)
                
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {str(e)}")
    
    def _process_orderbook(self, symbol: str, book_data: Dict):
        """Process orderbook updates"""
        try:
            for side in ['asks', 'bids']:
                if side in book_data:
                    for price, volume, timestamp in book_data[side]:
                        if float(volume) == 0:
                            self.orderbook[symbol][side].pop(float(price), None)
                        else:
                            self.orderbook[symbol][side][float(price)] = float(volume)
            
            # Trigger callbacks
            for callback in self.callbacks['book']:
                callback(symbol, self.orderbook[symbol])
                
        except Exception as e:
            logger.error(f"Error processing orderbook: {str(e)}")
    
    def add_callback(self, channel: str, callback: Callable):
        """Add callback function for specific channel"""
        self.callbacks[channel].append(callback)
    
    def start_processing(self):
        """Start message processing thread"""
        def process_messages():
            while self.running:
                try:
                    channel, symbol, data = self.message_queue.get(timeout=1)
                    if channel == 'trade':
                        self._process_trade(symbol, data)
                    elif channel == 'ohlc':
                        self._process_ohlc(symbol, data)
                    elif channel == 'book':
                        self._process_orderbook(symbol, data)
                except Queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
        
        self.processing_thread = threading.Thread(target=process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    async def run(self):
        """Main run loop with improved stability"""
        reconnect_delay = 1
        max_reconnect_delay = 300  # 5 minutes
        
        while True:
            try:
                if not self.ws or not self.running:
                    logger.info("Connecting to WebSocket...")
                    success = await self.connect()
                    if not success:
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        logger.warning(f"Connection failed, retrying in {reconnect_delay}s")
                        await asyncio.sleep(reconnect_delay)
                        continue
                    
                    # Successfully connected, reset reconnect delay
                    reconnect_delay = 1
                    
                    # Subscribe to channels
                    symbols = self.config.get_symbols()
                    channels = ["ticker", "ohlc", "trade"]
                    await self.subscribe(symbols, channels)
                
                # Process messages
                try:
                    async with asyncio.timeout(30):  # 30 seconds timeout
                        message = await self.ws.recv()
                        await self._handle_message(message)
                except asyncio.TimeoutError:
                    logger.warning("WebSocket read timeout, checking connection...")
                    if not await self.ensure_connection():
                        continue
                
                # Periodic health check
                if self.health_monitor.get_status()["last_message_age"] > 60:
                    logger.warning("No messages received recently, reconnecting...")
                    await self.ws.close()
                    self.ws = None
                    self.health_monitor.record_drop()
                    
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self.ws = None
                self.health_monitor.record_drop()
                await asyncio.sleep(reconnect_delay)
            except Exception as e:
                logger.error(f"Error in run loop: {str(e)}")
                await asyncio.sleep(reconnect_delay)
    
    def stop(self):
        """Stop the data stream"""
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
        logger.info("Data stream stopped")
    
    class ConnectionHealthMonitor:
        def __init__(self):
            self.last_message_time = time.time()
            self.connection_drops = 0
            self.message_count = 0
        
        def update(self):
            """Update on message received"""
            self.last_message_time = time.time()
            self.message_count += 1
        
        def record_drop(self):
            """Record connection drop"""
            self.connection_drops += 1
        
        def get_status(self):
            """Get connection health status"""
            return {
                "last_message_age": time.time() - self.last_message_time,
                "connection_drops": self.connection_drops,
                "message_count": self.message_count,
                "health": "good" if time.time() - self.last_message_time < 60 else "poor"
            }



# Update in reconnect logic
def update_reconnect_logic(self):
    self.health_monitor.record_drop()
    

