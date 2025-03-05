# src/trading/paper_trading.py

"""
Paper trading system for testing strategies without real money.
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

logger = log_manager.get_logger(__name__)

class PaperTradingExecutor:
    """Paper trading executor that simulates real trading"""
    
    def __init__(self):
        """Initialize paper trading system"""
        self.config = config.get_trading_params()
        self.balance = self.config.get('initial_balance', 10000)
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        self.equity_history = []
        
        # Last known prices
        self.latest_prices = {}
        
        # Initialize metrics
        self._record_equity()
    
    async def start(self):
        """Start the executor - required for compatibility with OrderExecutor"""
        logger.info("Paper trading executor started")
        return True

    async def stop(self):
        """Stop the executor - required for compatibility with OrderExecutor"""
        logger.info("Paper trading executor stopped")
        return True

    async def add_signal(self, signal):
        """Process a trading signal"""
        try:
            logger.info(f"Paper trading signal received: {signal}")
            
            # Extract signal details
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
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
                # Use the non-async version directly
                result = self._place_order_sync(order_data)
            elif action == 'sell':
                order_data['side'] = 'sell'
                # Use the non-async version directly
                result = self._place_order_sync(order_data)
            elif action == 'exit':
                # Find if we have any position for this symbol
                for symbol_pos, position in self.positions.items():
                    if symbol_pos == symbol:
                        # Close position
                        side = 'sell' if position['side'] == 'long' else 'buy'
                        order_data['side'] = side
                        # Use the non-async version directly
                        result = self._place_order_sync(order_data)
                        break
                else:
                    logger.warning(f"No position to exit for {symbol}")
                    return False
            
            logger.info(f"Paper trading signal processed: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing paper trading signal: {str(e)}")
            return False
            
    async def place_order(self, order_data):
        """Async wrapper for the synchronous place_order method"""
        return self._place_order_sync(order_data)

    def _place_order_sync(self, order_data):
        """Synchronous implementation of place order"""
        try:
            # Generate order ID
            order_id = f"paper_{int(time.time() * 1000)}_{len(self.orders)}"
            
            # Get order details
            symbol = order_data['symbol']
            side = order_data['side']
            order_type = order_data.get('type', 'market')
            price = order_data.get('price')
            size = order_data.get('size', 0)
            
            # For market orders, use latest price if available
            if order_type == 'market':
                if symbol in self.latest_prices:
                    price = self.latest_prices[symbol]
                elif not price:
                    return {'error': 'No price available for market order'}
            
            # Check if we have enough balance
            order_value = price * size
            
            if side == 'buy':
                if order_value > self.balance:
                    return {'error': 'Insufficient balance'}
            
            # Create order
            order = {
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'price': price,
                'size': size,
                'status': 'filled',  # Auto-fill for paper trading
                'created_at': datetime.now(),
                'filled_at': datetime.now()
            }
            
            # Add to orders
            self.orders[order_id] = order
            
            # Update balance and positions
            if side == 'buy':
                self.balance -= order_value
                
                # Add to position
                if symbol in self.positions:
                    # Average down
                    current_size = self.positions[symbol]['size']
                    current_price = self.positions[symbol]['entry_price']
                    new_size = current_size + size
                    new_price = (current_price * current_size + price * size) / new_size
                    
                    self.positions[symbol]['size'] = new_size
                    self.positions[symbol]['entry_price'] = new_price
                else:
                    # Create new position
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'size': size,
                        'entry_price': price,
                        'side': 'long',
                        'created_at': datetime.now()
                    }
            else:  # sell
                # Close existing position if any
                if symbol in self.positions and self.positions[symbol]['side'] == 'long':
                    position = self.positions[symbol]
                    
                    # Calculate P&L
                    entry_price = position['entry_price']
                    position_size = position['size']
                    
                    if size >= position_size:
                        # Close full position
                        close_size = position_size
                        remaining = size - position_size
                        
                        # Calculate P&L
                        pnl = (price - entry_price) * close_size
                        
                        # Update balance
                        self.balance += order_value + pnl
                        
                        # Record trade
                        trade = {
                            'symbol': symbol,
                            'side': 'long',
                            'entry_price': entry_price,
                            'exit_price': price,
                            'size': close_size,
                            'pnl': pnl,
                            'pnl_percent': (price / entry_price - 1) * 100,
                            'entry_time': position['created_at'],
                            'exit_time': datetime.now()
                        }
                        
                        self.trade_history.append(trade)
                        
                        # Remove position
                        del self.positions[symbol]
                        
                        # Create short position with remaining size if any
                        if remaining > 0:
                            self.positions[symbol] = {
                                'symbol': symbol,
                                'size': remaining,
                                'entry_price': price,
                                'side': 'short',
                                'created_at': datetime.now()
                            }
                    else:
                        # Partial close
                        close_size = size
                        remain_size = position_size - size
                        
                        # Calculate P&L
                        pnl = (price - entry_price) * close_size
                        
                        # Update balance
                        self.balance += order_value + pnl
                        
                        # Record trade
                        trade = {
                            'symbol': symbol,
                            'side': 'long',
                            'entry_price': entry_price,
                            'exit_price': price,
                            'size': close_size,
                            'pnl': pnl,
                            'pnl_percent': (price / entry_price - 1) * 100,
                            'entry_time': position['created_at'],
                            'exit_time': datetime.now()
                        }
                        
                        self.trade_history.append(trade)
                        
                        # Update position
                        self.positions[symbol]['size'] = remain_size
                else:
                    # Create new short position
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'size': size,
                        'entry_price': price,
                        'side': 'short',
                        'created_at': datetime.now()
                    }
                    
                    # Update balance
                    self.balance += order_value
            
            # Record new equity
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
    
    def update_price(self, symbol: str, price: float):
        """Update latest price for a symbol"""
        self.latest_prices[symbol] = price
        
        # Update unrealized P&L
        self._update_positions()
        
        # Record equity
        self._record_equity()
    
    def _update_positions(self):
        """Update position values with latest prices"""
        for symbol, position in self.positions.items():
            if symbol in self.latest_prices:
                current_price = self.latest_prices[symbol]
                entry_price = position['entry_price']
                size = position['size']
                side = position['side']
                
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                    
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                position['unrealized_pnl_percent'] = (
                    (current_price / entry_price - 1) * 100 if side == 'long' 
                    else (entry_price / current_price - 1) * 100
                )
    
    def _record_equity(self):
        """Record equity value"""
        # Calculate equity
        equity = self.balance
        
        # Add unrealized P&L
        for symbol, position in self.positions.items():
            if 'unrealized_pnl' in position:
                equity += position['unrealized_pnl']
                
        # Record
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
        return pd.DataFrame(self.equity_history)
    
    def get_ticker_sync(self, symbol):
        """Synchronous version of get_ticker"""
        if symbol in self.latest_prices:
            price = self.latest_prices[symbol]
        else:
            # Use a reasonable default price for testing
            price = 50000 if 'BTC' in symbol else 2000 if 'ETH' in symbol else 100
            self.latest_prices[symbol] = price
        
        return {
            'symbol': symbol,
            'last_price': price,
            'bid': price * 0.999,
            'ask': price * 1.001,
            'volume': 100,
            'timestamp': datetime.now()
        }
    
    async def get_ticker(self, symbol):
        """Async wrapper for get_ticker_sync"""
        return self.get_ticker_sync(symbol)
    
    def generate_mock_price_updates(self):
        """Generate mock price updates for testing"""
        import threading
        import random
        import time
        
        def update_prices():
            symbols = ['BTC/USD', 'ETH/USD']
            
            while True:
                for symbol in symbols:
                    # Get current price or set initial
                    if symbol in self.latest_prices:
                        current = self.latest_prices[symbol]
                    else:
                        current = 50000 if symbol == 'BTC/USD' else 2000
                    
                    # Random walk price
                    change_pct = random.uniform(-0.002, 0.002)  # 0.2% max change
                    new_price = current * (1 + change_pct)
                    
                    # Update price
                    self.update_price(symbol, new_price)
                
                # Sleep for 10 seconds
                time.sleep(10)
        
        # Start in a background thread
        price_thread = threading.Thread(target=update_prices, daemon=True)
        price_thread.start()
        logger.info("Started mock price update thread")