"""
Paper trading system for testing strategies without real money,
but using real market data.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import threading
import pandas as pd
import numpy as np

from ..core.logger import log_manager
from ..core.config import config
from ..data.stream import KrakenStreamManager

logger = log_manager.get_logger(__name__)

class PaperTradingExecutor:
    """Paper trading executor that connects to real market data"""
    
    def __init__(self):
        """Initialize paper trading system with real market data"""
        self.config = config.get_trading_params()
        self.balance = self.config.get('initial_balance', 10000)
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        self.equity_history = []
        
        # Last known prices from real market data
        self.latest_prices = {}
        
        # Stream manager for real data
        self.stream_manager = KrakenStreamManager()
        self.initialized = False
        
        # Initialize metrics
        self._record_equity()
    
    async def start(self):
        """Start the executor with real data connection"""
        try:
            # Connect to exchange
            logger.info("Connecting to exchange API for real market data...")
            connected = await self.stream_manager.connect()
            if not connected:
                logger.error("Failed to connect to exchange API")
                return False
            
            # Subscribe to market data for configured symbols
            symbols = config.get_symbols()
            channels = ["ticker", "ohlc"]  # Subscribe to ticker and candle data
            
            logger.info(f"Subscribing to {len(symbols)} symbols: {symbols}")
            await self.stream_manager.subscribe(symbols, channels)
            
            # Set up callback to update prices when new data arrives
            self.stream_manager.add_callback('ticker', self._update_ticker)
            self.stream_manager.add_callback('ohlc', self._update_ohlc)
            
            # Start the stream's message processing loop
            asyncio.create_task(self.stream_manager.run())
            
            self.initialized = True
            logger.info(f"Paper trading started with real market data for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error starting paper trading with real data: {str(e)}")
            return False
    
    async def stop(self):
        """Stop the executor and disconnect from API"""
        try:
            # Save final state
            self._record_equity()
            
            # Disconnect from API
            if hasattr(self.stream_manager, 'ws') and self.stream_manager.ws:
                self.stream_manager.running = False
                await self.stream_manager.ws.close()
            
            logger.info("Paper trading stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping paper trading: {str(e)}")
            return False
    
    def _update_ticker(self, symbol, ticker_data):
        """Update latest price from real ticker data"""
        try:
            # Extract price from ticker data - Kraken format varies by endpoint
            if isinstance(ticker_data, dict):
                if 'c' in ticker_data:  # Kraken spot format: c = last trade closed array
                    price = float(ticker_data['c'][0])
                elif 'last' in ticker_data:  # Common format
                    price = float(ticker_data['last'])
                elif 'lastPrice' in ticker_data:  # Kraken futures format
                    price = float(ticker_data['lastPrice'])
                else:
                    logger.warning(f"Unknown ticker format: {ticker_data}")
                    return
            elif isinstance(ticker_data, list) and len(ticker_data) > 0:
                # Some APIs return array format
                price = float(ticker_data[0])
            else:
                logger.warning(f"Could not extract price from ticker: {ticker_data}")
                return
            
            # Update latest price
            prev_price = self.latest_prices.get(symbol)
            self.latest_prices[symbol] = price
            
            # Log significant price changes (more than 0.5%)
            if prev_price and abs(price/prev_price - 1) > 0.005:
                logger.info(f"Price change: {symbol} from {prev_price:.2f} to {price:.2f} " 
                           f"({(price/prev_price - 1)*100:.2f}%)")
            
            # Update positions with latest price
            self._update_positions()
            
            # Record equity
            self._record_equity()
            
        except Exception as e:
            logger.error(f"Error updating ticker: {str(e)}")
    
    def _update_ohlc(self, symbol, ohlcv_data):
        """Update with OHLCV candle data"""
        try:
            # Kraken OHLCV format is typically an array 
            # [time, open, high, low, close, volume, ...]
            if isinstance(ohlcv_data, list) and len(ohlcv_data) >= 6:
                close_price = float(ohlcv_data[4])
                self.latest_prices[symbol] = close_price
                
                # Update positions and equity
                self._update_positions()
                self._record_equity()
                
        except Exception as e:
            logger.error(f"Error updating OHLCV: {str(e)}")
    
    async def add_signal(self, signal):
        """Process a trading signal"""
        try:
            logger.info(f"Paper trading signal received: {signal}")
            
            # Extract signal details
            symbol = signal['symbol']
            action = signal['action']
            
            # Ensure we have a current price
            if symbol not in self.latest_prices:
                logger.warning(f"No current price available for {symbol}")
                return False
            
            # Use the current market price
            price = self.latest_prices[symbol]
            size = signal.get('size', 1.0)
            
            # Create order data
            order_data = {
                'symbol': symbol,
                'price': price,
                'size': size
            }
            
            # Process based on action
            if action == 'buy':
                order_data['side'] = 'buy'
                result = self._place_order_sync(order_data)
            elif action == 'sell':
                order_data['side'] = 'sell'
                result = self._place_order_sync(order_data)
            elif action == 'exit':
                # Find if we have any position for this symbol
                if symbol in self.positions:
                    # Close position
                    side = 'sell' if self.positions[symbol]['side'] == 'long' else 'buy'
                    order_data['side'] = side
                    result = self._place_order_sync(order_data)
                else:
                    logger.warning(f"No position to exit for {symbol}")
                    return False
            
            logger.info(f"Paper trading signal processed: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing paper trading signal: {str(e)}")
            return False
    
    def _place_order_sync(self, order_data):
        """Process a trading order in the paper trading environment"""
        try:
            # Generate order ID
            order_id = f"paper_{int(time.time() * 1000)}_{len(self.orders)}"
            
            # Get order details
            symbol = order_data['symbol']
            side = order_data['side']
            order_type = order_data.get('type', 'market')
            price = order_data.get('price')
            size = order_data.get('size', 0)
            
            # Add slippage for market orders
            if order_type == 'market':
                slippage = self.config.get('slippage_factor', 0.001)  # Default to 0.1%
                if side == 'buy':
                    price *= (1 + slippage)  # Higher price for buys
                else:
                    price *= (1 - slippage)  # Lower price for sells
            
            # Calculate fees
            fee_rate = self.config.get('transaction_fee_rate', 0.0026)  # Default to 0.26%
            fee = price * size * fee_rate
            
            # Check if we have enough balance for buys
            if side == 'buy':
                total_cost = price * size + fee
                if total_cost > self.balance:
                    logger.warning(f"Insufficient balance: {self.balance} < {total_cost}")
                    return {'error': 'Insufficient balance'}
                
                # Deduct from balance
                self.balance -= total_cost
                
                # Update position
                self._add_to_position(symbol, price, size, 'long')
            else:  # sell
                # Calculate position value
                position_value = price * size
                net_proceeds = position_value - fee
                
                # Update position
                self._reduce_position(symbol, price, size, 'long')
                
                # Add to balance
                self.balance += net_proceeds
            
            # Create order record
            order = {
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'price': price,
                'size': size,
                'fee': fee,
                'status': 'filled',
                'created_at': datetime.now(),
                'filled_at': datetime.now()
            }
            
            # Add to orders
            self.orders[order_id] = order
            
            # Record equity
            self._record_equity()
            
            logger.info(f"Paper trade executed: {side} {size} {symbol} @ {price}")
            
            return {
                'order_id': order_id,
                'status': 'filled',
                'filled_price': price,
                'filled_size': size
            }
            
        except Exception as e:
            logger.error(f"Error in paper trading: {str(e)}")
            return {'error': str(e)}
    
    def _add_to_position(self, symbol, price, size, side):
        """Add to or create a position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if position['side'] == side:
                # Average down/up
                current_size = position['size']
                current_price = position['entry_price']
                new_size = current_size + size
                new_price = (current_price * current_size + price * size) / new_size
                
                position['size'] = new_size
                position['entry_price'] = new_price
            else:
                # Close opposite position
                current_size = position['size']
                
                if size >= current_size:
                    # Close entire position and flip
                    remaining = size - current_size
                    
                    # Calculate P&L from closing
                    entry_price = position['entry_price']
                    exit_price = price
                    
                    if position['side'] == 'long':
                        pnl = (exit_price - entry_price) * current_size
                    else:
                        pnl = (entry_price - exit_price) * current_size
                    
                    # Record trade
                    self._record_trade(symbol, position['side'], entry_price, exit_price, 
                                      current_size, pnl, position['entry_time'])
                    
                    # Create new position if there's remaining size
                    if remaining > 0:
                        self.positions[symbol] = {
                            'symbol': symbol,
                            'side': side,
                            'size': remaining,
                            'entry_price': price,
                            'entry_time': datetime.now(),
                            'current_price': price,
                            'unrealized_pnl': 0
                        }
                    else:
                        # Just remove the position
                        del self.positions[symbol]
                else:
                    # Reduce position
                    position['size'] -= size
                    
                    # Calculate P&L
                    entry_price = position['entry_price']
                    exit_price = price
                    
                    if position['side'] == 'long':
                        pnl = (exit_price - entry_price) * size
                    else:
                        pnl = (entry_price - exit_price) * size
                    
                    # Record trade
                    self._record_trade(symbol, position['side'], entry_price, exit_price, 
                                      size, pnl, position['entry_time'])
        else:
            # Create new position
            self.positions[symbol] = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': price,
                'entry_time': datetime.now(),
                'current_price': price,
                'unrealized_pnl': 0
            }
    
    def _reduce_position(self, symbol, price, size, expected_side):
        """Reduce or close a position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Check if position side matches
            if position['side'] != expected_side:
                logger.warning(f"Position side mismatch: {position['side']} != {expected_side}")
                # Handle unexpected side (could be a short sale or other situation)
                self._add_to_position(symbol, price, size, 'short')
                return
            
            current_size = position['size']
            
            if size >= current_size:
                # Close entire position
                entry_price = position['entry_price']
                exit_price = price
                
                if position['side'] == 'long':
                    pnl = (exit_price - entry_price) * current_size
                else:
                    pnl = (entry_price - exit_price) * current_size
                
                # Record trade
                self._record_trade(symbol, position['side'], entry_price, exit_price,
                                  current_size, pnl, position['entry_time'])
                
                # Remove position
                del self.positions[symbol]
            else:
                # Partial close
                entry_price = position['entry_price']
                exit_price = price
                
                if position['side'] == 'long':
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size
                
                # Record trade
                self._record_trade(symbol, position['side'], entry_price, exit_price,
                                  size, pnl, position['entry_time'])
                
                # Update position
                position['size'] -= size
        else:
            logger.warning(f"No position found for {symbol} to reduce")
            # Create a new position of the opposite side
            opposite_side = 'short' if expected_side == 'long' else 'long'
            self._add_to_position(symbol, price, size, opposite_side)
    
    def _record_trade(self, symbol, side, entry_price, exit_price, size, pnl, entry_time):
        """Record a completed trade in the trade history"""
        trade = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_percent': (
                (exit_price / entry_price - 1) * 100 if side == 'long' 
                else (entry_price / exit_price - 1) * 100
            ),
            'entry_time': entry_time,
            'exit_time': datetime.now()
        }
        
        self.trade_history.append(trade)
        logger.info(f"Completed trade: {symbol} {side} {size} @ {exit_price}, PnL: ${pnl:.2f}")
        
        # Store trade in database if available
        try:
            from ..data.database import db
            db_trade = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'amount': size,
                'entry_time': entry_time,
                'exit_time': datetime.now(),
                'pnl': pnl,
                'status': 'closed',
                'strategy': 'paper_trade'
            }
            db.store_trade(db_trade)
        except Exception as e:
            logger.warning(f"Could not store trade in database: {str(e)}")
    
    def _update_positions(self):
        """Update positions with latest prices"""
        for symbol, position in list(self.positions.items()):
            if symbol in self.latest_prices:
                current_price = self.latest_prices[symbol]
                entry_price = position['entry_price']
                size = position['size']
                side = position['side']
                
                # Calculate unrealized P&L
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                
                # Update position
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                position['unrealized_pnl_percent'] = (
                    (current_price / entry_price - 1) * 100 if side == 'long' 
                    else (entry_price / current_price - 1) * 100
                )
                
                # Check for stop loss or take profit
                self._check_stop_loss_take_profit(symbol, position)
    
    def _check_stop_loss_take_profit(self, symbol, position):
        """Check if position should be closed due to stop loss or take profit"""
        # Get stop loss and take profit parameters
        stop_loss_pct = self.config.get('stop_loss', 0.05)  # 5% default
        take_profit_pct = self.config.get('profit_target', 0.1)  # 10% default
        
        side = position['side']
        entry_price = position['entry_price']
        current_price = position['current_price']
        
        # Calculate stop and target prices
        if side == 'long':
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + take_profit_pct)
            
            # Check if triggered
            if current_price <= stop_price:
                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                # Create exit order
                order_data = {
                    'symbol': symbol,
                    'side': 'sell',
                    'price': current_price,
                    'size': position['size'],
                    'type': 'market'
                }
                self._place_order_sync(order_data)
            
            elif current_price >= target_price:
                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                # Create exit order
                order_data = {
                    'symbol': symbol,
                    'side': 'sell',
                    'price': current_price,
                    'size': position['size'],
                    'type': 'market'
                }
                self._place_order_sync(order_data)
                
        else:  # short
            stop_price = entry_price * (1 + stop_loss_pct)
            target_price = entry_price * (1 - take_profit_pct)
            
            # Check if triggered
            if current_price >= stop_price:
                logger.info(f"Stop loss triggered for {symbol} short at {current_price}")
                # Create exit order
                order_data = {
                    'symbol': symbol,
                    'side': 'buy',
                    'price': current_price,
                    'size': position['size'],
                    'type': 'market'
                }
                self._place_order_sync(order_data)
            
            elif current_price <= target_price:
                logger.info(f"Take profit triggered for {symbol} short at {current_price}")
                # Create exit order
                order_data = {
                    'symbol': symbol,
                    'side': 'buy',
                    'price': current_price,
                    'size': position['size'],
                    'type': 'market'
                }
                self._place_order_sync(order_data)
    
    def _record_equity(self):
        """Record current equity value"""
        equity = self.balance
        
        # Add unrealized P&L from positions
        for symbol, position in self.positions.items():
            if 'unrealized_pnl' in position:
                equity += position['unrealized_pnl']
        
        # Add to equity history
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': equity
        })
    
    def get_account_balance(self) -> Dict[str, Any]:
        """Get current account balance"""
        total_equity = self.balance
        unrealized_pnl = 0
        
        for position in self.positions.values():
            if 'unrealized_pnl' in position:
                unrealized_pnl += position['unrealized_pnl']
        
        total_equity += unrealized_pnl
        
        return {
            'balance': self.balance,
            'equity': total_equity,
            'unrealized_pnl': unrealized_pnl
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        return list(self.positions.values())
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history
    
    def get_equity_history(self) -> pd.DataFrame:
        """Get equity history as DataFrame"""
        if not self.equity_history:
            return pd.DataFrame(columns=['timestamp', 'equity'])
        return pd.DataFrame(self.equity_history)
    
    async def get_ticker(self, symbol):
        """Get current ticker for a symbol"""
        # Check if we have a recorded price
        if symbol in self.latest_prices:
            price = self.latest_prices[symbol]
            return {
                'symbol': symbol,
                'last_price': price,
                'bid': price * 0.999,  # Simulate bid/ask spread
                'ask': price * 1.001,
                'volume': 100,  # Placeholder
                'timestamp': datetime.now()
            }
        
        # If not initialized with real data or no price yet, fetch from API
        if not self.initialized:
            # Create a fake ticker with reasonable defaults
            price = 50000 if 'BTC' in symbol else 2000 if 'ETH' in symbol else 100
            return {
                'symbol': symbol,
                'last_price': price,
                'bid': price * 0.999,
                'ask': price * 1.001,
                'volume': 100,
                'timestamp': datetime.now()
            }
        
        # Try to get from the stream manager directly
        try:
            ticker = await self.stream_manager.get_ticker(symbol)
            if ticker:
                # Store for future use
                if 'last_price' in ticker:
                    self.latest_prices[symbol] = ticker['last_price']
                return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker: {str(e)}")
        
        # Fallback to default values
        price = 50000 if 'BTC' in symbol else 2000 if 'ETH' in symbol else 100
        return {
            'symbol': symbol,
            'last_price': price,
            'bid': price * 0.999,
            'ask': price * 1.001,
            'volume': 100,
            'timestamp': datetime.now()
        }