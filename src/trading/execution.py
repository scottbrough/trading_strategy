# src/trading/execution.py
"""
Order execution system for the trading platform.
Handles converting strategy signals to exchange orders with proper risk management.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.logger import log_manager
from ..core.config import config
from ..core.exceptions import ExecutionError
from ..data.stream import KrakenStreamManager
from ..strategy.risk_management import RiskManager

logger = log_manager.get_logger(__name__)

class OrderExecutor:
    def __init__(self, stream_manager: KrakenStreamManager, risk_manager: RiskManager):
        """Initialize order executor with stream and risk managers"""
        self.stream_manager = stream_manager
        self.risk_manager = risk_manager
        self.orders = {}
        self.positions = {}
        self.pending_signals = []
        self.execution_thread = None
        self.running = False
        self.last_execution_time = {}  # Track last execution time by symbol
        
    async def start(self):
        """Start the order execution loop"""
        if self.execution_thread and self.execution_thread.is_alive():
            logger.warning("Execution loop already running")
            return False
            
        self.running = True
        self.execution_thread = asyncio.create_task(self._execution_loop())
        logger.info("Order execution system started")
        return True
        
    async def stop(self):
        """Stop the order execution loop"""
        self.running = False
        if self.execution_thread:
            await self.execution_thread
        logger.info("Order execution system stopped")
        
    async def add_signal(self, signal: Dict[str, Any]):
        """Add a trading signal to the execution queue"""
        try:
            # Validate signal
            required_fields = ['timestamp', 'symbol', 'action', 'price']
            if not all(field in signal for field in required_fields):
                logger.error(f"Invalid signal, missing required fields: {signal}")
                return False
                
            # Add to pending signals
            self.pending_signals.append(signal)
            logger.info(f"Added signal: {signal['action']} {signal['symbol']} @ {signal['price']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding signal: {str(e)}")
            return False
            
    async def _execution_loop(self):
        """Main execution loop"""
        while self.running:
            try:
                # Process any pending signals
                if self.pending_signals:
                    signal = self.pending_signals.pop(0)
                    await self._process_signal(signal)
                    
                # Monitor open positions
                await self._monitor_positions()
                
                # Sleep to avoid excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {str(e)}")
                await asyncio.sleep(1)
                
    async def _process_signal(self, signal: Dict[str, Any]):
        """Process a single trading signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            
            # Check if we should rate limit this symbol
            current_time = time.time()
            min_interval = self.config.get('min_execution_interval', 10)  # seconds
            
            if symbol in self.last_execution_time:
                time_since_last = current_time - self.last_execution_time[symbol]
                if time_since_last < min_interval:
                    logger.info(f"Rate limiting execution for {symbol}, "
                               f"only {time_since_last:.1f}s since last execution")
                    self.pending_signals.append(signal)  # Put back in queue
                    return
            
            # Update last execution time
            self.last_execution_time[symbol] = current_time
            
            # Handle different actions
            if action == 'buy':
                await self._execute_buy(signal)
            elif action == 'sell':
                await self._execute_sell(signal)
            elif action == 'exit':
                await self._execute_exit(signal)
            else:
                logger.warning(f"Unknown action in signal: {action}")
                
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            
    async def _execute_buy(self, signal: Dict[str, Any]):
        """Execute a buy signal"""
        try:
            symbol = signal['symbol']
            price = signal['price']
            
            # Check risk limits
            if not self.risk_manager.check_risk_limits({'symbol': symbol, 'action': 'buy', 'price': price}):
                logger.warning(f"Risk limits prevent buy for {symbol}")
                return
                
            # Calculate position size
            balance = await self.stream_manager.get_account_balance()
            size = self.risk_manager.calculate_position_size(
                balance,
                price,
                signal.get('volatility', 0.1),
                {'symbol': symbol}
            )
            
            # Apply minimum position size
            min_size = config.get('min_position_size', 0.001)
            if size < min_size:
                logger.warning(f"Calculated position size too small: {size}, using minimum: {min_size}")
                size = min_size
                
            # Place order
            order = await self.stream_manager.market_buy(symbol, size)
            
            if order and order.get('order_id'):
                logger.info(f"Executed buy for {symbol}: {size} @ {price}")
                
                # Add to positions
                self.positions[symbol] = {
                    'side': 'long',
                    'size': size,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'order_id': order['order_id']
                }
                
            else:
                logger.error(f"Failed to execute buy for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing buy: {str(e)}")
            
    async def _execute_sell(self, signal: Dict[str, Any]):
        """Execute a sell signal"""
        try:
            symbol = signal['symbol']
            price = signal['price']
            
            # Check risk limits
            if not self.risk_manager.check_risk_limits({'symbol': symbol, 'action': 'sell', 'price': price}):
                logger.warning(f"Risk limits prevent sell for {symbol}")
                return
                
            # Calculate position size
            balance = await self.stream_manager.get_account_balance()
            size = self.risk_manager.calculate_position_size(
                balance,
                price,
                signal.get('volatility', 0.1),
                {'symbol': symbol}
            )
            
            # Apply minimum position size
            min_size = config.get('min_position_size', 0.001)
            if size < min_size:
                logger.warning(f"Calculated position size too small: {size}, using minimum: {min_size}")
                size = min_size
                
            # Place order
            order = await self.stream_manager.market_sell(symbol, size)
            
            if order and order.get('order_id'):
                logger.info(f"Executed sell for {symbol}: {size} @ {price}")
                
                # Add to positions
                self.positions[symbol] = {
                    'side': 'short',
                    'size': size,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'order_id': order['order_id']
                }
                
            else:
                logger.error(f"Failed to execute sell for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing sell: {str(e)}")
            
    async def _execute_exit(self, signal: Dict[str, Any]):
        """Execute an exit signal"""
        try:
            symbol = signal['symbol']
            price = signal['price']
            
            # Check if we have a position to exit
            if symbol not in self.positions:
                logger.warning(f"No position to exit for {symbol}")
                return
                
            position = self.positions[symbol]
            side = position['side']
            size = position['size']
            
            # Execute appropriate exit order
            if side == 'long':
                order = await self.stream_manager.market_sell(symbol, size)
            else:  # short
                order = await self.stream_manager.market_buy(symbol, size)
                
            if order and order.get('order_id'):
                logger.info(f"Executed exit for {symbol} {side} position: {size} @ {price}")
                
                # Calculate P&L
                entry_price = position['entry_price']
                if side == 'long':
                    pnl = (price - entry_price) * size
                else:  # short
                    pnl = (entry_price - price) * size
                    
                # Record trade and remove position
                trade = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'size': size,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'pnl': pnl,
                    'pnl_pct': (pnl / (entry_price * size)) * 100
                }
                
                # Log trade and update risk manager
                logger.info(f"Trade completed: {trade}")
                self.risk_manager.log_trade(trade)
                
                # Remove position
                del self.positions[symbol]
                
            else:
                logger.error(f"Failed to execute exit for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing exit: {str(e)}")
            
    async def _monitor_positions(self):
        """Monitor open positions for stop loss or take profit conditions"""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                ticker = await self.stream_manager.get_ticker(symbol)
                if not ticker:
                    continue
                    
                current_price = ticker['last_price']
                position['current_price'] = current_price
                
                # Calculate unrealized P&L
                entry_price = position['entry_price']
                size = position['size']
                side = position['side']
                
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                    
                position['unrealized_pnl'] = unrealized_pnl
                
                # Check for stop loss or take profit conditions
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')
                
                if stop_loss:
                    if (side == 'long' and current_price <= stop_loss) or \
                       (side == 'short' and current_price >= stop_loss):
                        logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                        await self._execute_exit({'symbol': symbol, 'price': current_price, 'action': 'exit'})
                        continue
                        
                if take_profit:
                    if (side == 'long' and current_price >= take_profit) or \
                       (side == 'short' and current_price <= take_profit):
                        logger.info(f"Take profit triggered for {symbol} at {current_price}")
                        await self._execute_exit({'symbol': symbol, 'price': current_price, 'action': 'exit'})
                        continue
                
            except Exception as e:
                logger.error(f"Error monitoring position for {symbol}: {str(e)}")