"""
Momentum trading strategy implementation.
Uses multiple technical indicators to identify momentum opportunities.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..base import BaseStrategy
from ..indicators import TechnicalIndicators
from ...core.logger import log_manager

logger = log_manager.get_logger(__name__)

class MomentumStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.indicators = TechnicalIndicators()
        self.required_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'adx', 'atr', 'slowk', 'slowd'
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """
        Generate trading signals based on momentum indicators
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            List of signal dictionaries
        """
        try:
            signals = []
            
            # Process data with indicators if needed
            if not all(ind in data.columns for ind in self.required_indicators):
                data = self.indicators.calculate_all(data)
            
            # Generate signals for each candle
            for i in range(1, len(data)):
                current = data.iloc[i]
                prev = data.iloc[i-1]
                
                # Calculate signal strength and conditions
                signal_strength = self._calculate_signal_strength(current)
                entry_conditions = self._check_entry_conditions(current, prev)
                exit_conditions = self._check_exit_conditions(current, prev)
                
                # Process entry signals
                if entry_conditions['should_enter']:
                    signal = {
                        'timestamp': current.name,
                        'action': 'buy' if entry_conditions['direction'] == 'long' else 'sell',
                        'price': current['close'],
                        'size': self._calculate_position_size(current, signal_strength),
                        'type': 'market',
                        'reason': entry_conditions['reason'],
                        'parameters': {
                            'signal_strength': signal_strength,
                            'rsi': current['rsi'],
                            'macd_hist': current['macd_hist'],
                            'adx': current['adx']
                        }
                    }
                    signals.append(signal)
                
                # Process exit signals
                elif exit_conditions['should_exit']:
                    signal = {
                        'timestamp': current.name,
                        'action': 'exit',
                        'price': current['close'],
                        'type': 'market',
                        'reason': exit_conditions['reason']
                    }
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def _calculate_signal_strength(self, candle: pd.Series) -> float:
        """Calculate momentum signal strength"""
        try:
            # RSI component
            rsi_strength = (
                (candle['rsi'] - 30) / 20 if candle['rsi'] < 50
                else (70 - candle['rsi']) / 20
            )
            
            # MACD component
            macd_strength = abs(candle['macd_hist']) / candle['close']
            
            # ADX component
            adx_strength = (candle['adx'] - 20) / 30 if candle['adx'] > 20 else 0
            
            # Combine components
            strength = (
                0.4 * rsi_strength +
                0.4 * macd_strength +
                0.2 * adx_strength
            )
            
            return max(0, min(strength, 1))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0
    
    def _check_entry_conditions(self, current: pd.Series, prev: pd.Series) -> Dict:
        """Check entry conditions"""
        try:
            # Initialize result
            result = {
                'should_enter': False,
                'direction': None,
                'reason': None
            }
            
            # Check long entry conditions
            long_conditions = [
                current['rsi'] < self.params['rsi_oversold'],
                current['macd_hist'] > prev['macd_hist'],
                current['adx'] > self.params['adx_threshold'],
                current['slowk'] > current['slowd']
            ]
            
            # Check short entry conditions
            short_conditions = [
                current['rsi'] > self.params['rsi_overbought'],
                current['macd_hist'] < prev['macd_hist'],
                current['adx'] > self.params['adx_threshold'],
                current['slowk'] < current['slowd']
            ]
            
            if all(long_conditions):
                result.update({
                    'should_enter': True,
                    'direction': 'long',
                    'reason': 'momentum_long'
                })
            elif all(short_conditions):
                result.update({
                    'should_enter': True,
                    'direction': 'short',
                    'reason': 'momentum_short'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking entry conditions: {str(e)}")
            return {'should_enter': False, 'direction': None, 'reason': None}
    
    def _check_exit_conditions(self, current: pd.Series, prev: pd.Series) -> Dict:
        """Check exit conditions"""
        try:
            # Initialize result
            result = {
                'should_exit': False,
                'reason': None
            }
            
            # Exit conditions for long positions
            long_exit_conditions = [
                current['rsi'] > 70,
                current['macd_hist'] < 0,
                current['slowk'] < current['slowd']
            ]
            
            # Exit conditions for short positions
            short_exit_conditions = [
                current['rsi'] < 30,
                current['macd_hist'] > 0,
                current['slowk'] > current['slowd']
            ]
            
            if any([all(long_exit_conditions), all(short_exit_conditions)]):
                result.update({
                    'should_exit': True,
                    'reason': 'momentum_reversal'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            return {'should_exit': False, 'reason': None}
    
    def _calculate_position_size(self, candle: pd.Series, signal_strength: float, current_capital: float) -> float:
        """Calculate position size based on signal strength, ATR, and CURRENT capital"""
        try:
            # Use CURRENT capital instead of initial capital
            base_size_usd = self.params['risk_factor'] * current_capital
            
            # Convert to quantity based on current price
            base_quantity = base_size_usd / candle['close']
            
            # Adjust for signal strength
            quantity = base_quantity * signal_strength
            
            # Adjust for volatility using ATR
            atr_factor = 1.0 - (candle['atr'] / candle['close'])
            quantity *= max(0.2, min(atr_factor, 1.0))
            
            # Apply position limits as QUANTITIES
            max_quantity = (self.params['max_position_size'] * current_capital) / candle['close']
            min_quantity = (self.params['min_position_size'] * current_capital) / candle['close']
            
            return max(min_quantity, min(quantity, max_quantity))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0