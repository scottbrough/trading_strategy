"""
Base strategy class for the trading system.
Provides foundation for implementing trading strategies with standard interfaces.
"""

import abc
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.logger import log_manager
from ..core.exceptions import StrategyError

logger = log_manager.get_logger(__name__)

class BaseStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration including parameters and risk settings
        """
        self.config = config
        self.params = config.get("strategy", {})
        self.positions = []
        self.trades = []
        self.last_update = None
        self._validate_config()
    
    def _validate_config(self):
        """Validate strategy configuration"""
        required_params = [
            'risk_fraction',
            'max_position_size',
            'stop_loss',
            'profit_target'
        ]
        
        missing_params = [p for p in required_params if p not in self.params]
        if missing_params:
            print("DEBUG: Configuration received:")
            print(f"self.config = {self.config}")
            print(f"self.params = {self.params}")
            print(f"Missing parameters: {missing_params}")
            raise StrategyError.ValidationError(
                "Missing required parameters",
                details={'missing': missing_params}
            )
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """
        Generate trading signals from market data.
        Must be implemented by concrete strategies.
        
        Args:
            data: DataFrame containing processed market data
            
        Returns:
            List of signal dictionaries with entries like:
            {
                'timestamp': datetime,
                'action': 'buy'/'sell'/'exit',
                'price': float,
                'size': float,
                'type': 'market'/'limit',
                'reason': str,
                'parameters': dict
            }
        """
        pass
    
    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal dictionary to validate
            
        Returns:
            bool: True if signal is valid
        """
        try:
            required_fields = ['timestamp', 'action', 'price', 'size']
            
            # Check required fields
            if not all(field in signal for field in required_fields):
                return False
            
            # Validate action
            if signal['action'] not in ['buy', 'sell', 'exit']:
                return False
            
            # Validate price and size
            if not (isinstance(signal['price'], (int, float)) and signal['price'] > 0):
                return False
            if not (isinstance(signal['size'], (int, float)) and signal['size'] > 0):
                return False
            
            # Validate timestamp
            if not isinstance(signal['timestamp'], datetime):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation error: {str(e)}")
            return False
    
    def calculate_position_size(self, 
                              capital: float,
                              price: float,
                              risk_metrics: Dict) -> float:
        """
        Calculate position size based on capital and risk parameters.
        
        Args:
            capital: Available capital
            price: Current price
            risk_metrics: Dictionary containing risk metrics
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Get risk parameters
            risk_fraction = self.params['risk_fraction']
            max_position = self.params['max_position_size']
            
            # Calculate based on risk fraction
            risk_amount = capital * risk_fraction
            
            # Adjust for volatility if available
            if 'volatility' in risk_metrics:
                vol_factor = 1.0 - (min(risk_metrics['volatility'], 50) / 100)
                risk_amount *= vol_factor
            
            # Calculate position size
            position_size = risk_amount / price
            
            # Apply position limits
            max_size = (capital * max_position) / price
            position_size = min(position_size, max_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Position size calculation error: {str(e)}")
            return 0.0
    
    def update_positions(self, current_data: pd.DataFrame):
        """
        Update existing positions with current market data.
        
        Args:
            current_data: DataFrame with current market data
        """
        try:
            current_price = current_data['close'].iloc[-1]
            current_time = current_data.index[-1]
            
            for position in self.positions:
                # Update unrealized P&L
                price_diff = current_price - position['entry_price']
                multiplier = 1 if position['side'] == 'buy' else -1
                position['unrealized_pnl'] = price_diff * position['size'] * multiplier
                
                # Check stop loss
                stop_price = position['entry_price'] * (
                    1 - self.params['stop_loss'] if position['side'] == 'buy'
                    else 1 + self.params['stop_loss']
                )
                
                # Check take profit
                profit_price = position['entry_price'] * (
                    1 + self.params['profit_target'] if position['side'] == 'buy'
                    else 1 - self.params['profit_target']
                )
                
                # Check exit conditions
                if (position['side'] == 'buy' and current_price <= stop_price) or \
                   (position['side'] == 'sell' and current_price >= stop_price):
                    self._close_position(position, current_price, current_time, 'stop_loss')
                
                elif (position['side'] == 'buy' and current_price >= profit_price) or \
                     (position['side'] == 'sell' and current_price <= profit_price):
                    self._close_position(position, current_price, current_time, 'take_profit')
            
        except Exception as e:
            logger.error(f"Position update error: {str(e)}")
    
    def _close_position(self, position: Dict, price: float, time: datetime, reason: str):
        """Close a position and record the trade"""
        try:
            # Calculate realized P&L
            price_diff = price - position['entry_price']
            multiplier = 1 if position['side'] == 'buy' else -1
            realized_pnl = price_diff * position['size'] * multiplier
            
            # Record trade
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': time,
                'symbol': position['symbol'],
                'side': position['side'],
                'size': position['size'],
                'entry_price': position['entry_price'],
                'exit_price': price,
                'realized_pnl': realized_pnl,
                'reason': reason
            }
            self.trades.append(trade)
            
            # Remove position
            self.positions.remove(position)
            
            logger.info(f"Closed position: {trade}")
            
        except Exception as e:
            logger.error(f"Position closing error: {str(e)}")
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions"""
        try:
            total_exposure = sum(p['size'] * p['entry_price'] for p in self.positions)
            total_pnl = sum(p['unrealized_pnl'] for p in self.positions)
            
            return {
                'num_positions': len(self.positions),
                'total_exposure': total_exposure,
                'total_pnl': total_pnl,
                'positions': self.positions
            }
            
        except Exception as e:
            logger.error(f"Position summary error: {str(e)}")
            return {}
    
    def get_trade_statistics(self) -> Dict:
        """Calculate trade statistics"""
        try:
            if not self.trades:
                return {}
            
            # Calculate basic statistics
            total_trades = len(self.trades)
            profitable_trades = sum(1 for t in self.trades if t['realized_pnl'] > 0)
            win_rate = profitable_trades / total_trades
            
            # Calculate P&L statistics
            total_pnl = sum(t['realized_pnl'] for t in self.trades)
            avg_profit = sum(t['realized_pnl'] for t in self.trades if t['realized_pnl'] > 0) / profitable_trades if profitable_trades > 0 else 0
            avg_loss = sum(t['realized_pnl'] for t in self.trades if t['realized_pnl'] < 0) / (total_trades - profitable_trades) if (total_trades - profitable_trades) > 0 else 0
            
            return {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'average_profit': avg_profit,
                'average_loss': avg_loss,
                'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Trade statistics error: {str(e)}")
            return {}