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
    def __init__(self):
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
                self.message_queue.put((channel_name, symbol, data[1]))
                
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
        for trade in trade_data:
            trade_record = {
                'price': float(trade[0]),
                'volume': float(trade[1]),
                'time': datetime.fromtimestamp(float(trade[2])),
                'side': 'buy' if trade[3] == 'b' else 'sell',
                'type': 'market' if trade[4] == 'm' else 'limit'
            }
            self.trades[symbol].append(trade_record)
            
            # Trigger callbacks
            for callback in self.callbacks['trade']:
                callback(symbol, trade_record)
    
    def _process_ohlc(self, symbol: str, ohlc_data: List):
        """Process OHLCV data"""
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
    
    def _process_orderbook(self, symbol: str, book_data: Dict):
        """Process orderbook updates"""
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
        """Main run loop"""
        while True:
            try:
                if not self.ws or not self.running:
                    success = await self.connect()
                    if not success:
                        await asyncio.sleep(self.reconnect_delay)
                        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                        continue
                    self.reconnect_delay = 1
                
                async for message in self.ws:
                    await self._handle_message(message)
                
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting to reconnect...")
                self.ws = None
            except Exception as e:
                logger.error(f"Error in run loop: {str(e)}")
                await asyncio.sleep(self.reconnect_delay)
    
    def stop(self):
        """Stop the data stream"""
        self.running = False
        if self.ws:
            asyncio.create_task(self.ws.close())
        logger.info("Data stream stopped")

# Example usage:
if __name__ == "__main__":
    stream = KrakenStreamManager()
    
    def print_trade(symbol: str, trade: dict):
        print(f"Trade: {symbol} - {trade}")
    
    def print_candle(symbol: str, candle: dict):
        print(f"Candle: {symbol} - {candle}")
    
    async def main():
        # Add callbacks
        stream.add_callback('trade', print_trade)
        stream.add_callback('ohlc', print_candle)
        
        # Start processing thread
        stream.start_processing()
        
        # Connect and subscribe
        await stream.connect()
        await stream.subscribe(['BTC/USD'], ['trade', 'ohlc'])
        
        # Run the stream
        await stream.run()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stream.stop()