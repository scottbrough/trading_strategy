"""
Momentum trading strategy implementation.
Uses multiple technical indicators to identify momentum opportunities
with improved signal quality and risk management.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
            'adx', 'atr', 'slowk', 'slowd', 'ema_9', 
            'ema_21', 'ema_50', 'volume_ratio'
        ]
        # Track last trade time for frequency control
        self.last_signal_time = None
        self.trades_today = 0
        self.current_day = None
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """
        Generate trading signals based on momentum indicators with improved quality
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            List of signal dictionaries
        """
        try:
            signals = []
            missing_indicators = [ind for ind in self.required_indicators if ind not in data.columns]
            if missing_indicators:
                logger.warning(f"Missing required indicators: {missing_indicators}")
                        # Process data with indicators if needed
            if not all(ind in data.columns for ind in self.required_indicators):
                data = self.indicators.calculate_all(data)
            
            # Add market regime detection
            data['regime'] = self._detect_market_regime(data)
            
            # Reset trade counter on new day
            self._reset_daily_counters(data)
            
            # Generate signals for each candle
            for i in range(2, len(data)):  # Need at least 2 previous candles
                current = data.iloc[i]
                prev = data.iloc[i-1]
                prev2 = data.iloc[i-2]
                
                # Skip processing if we've reached daily trade limit
                if self._trade_limit_reached(current):
                    continue
                
                # Skip if not in a trending market regime
                if current['regime'] != 'trending' and self.params.get('require_trend_confirmation', True):
                    continue
                
                # Calculate signal strength and conditions
                signal_strength = self._calculate_signal_strength(current, prev, prev2)
                entry_conditions = self._check_entry_conditions(current, prev)
                exit_conditions = self._check_exit_conditions(current, prev)
                
                # Process entry signals
                if entry_conditions['should_enter']:
                    # Calculate appropriate position size
                    size = self._calculate_position_size(current, signal_strength)
                    
                    signal = {
                        'timestamp': current.name,
                        'action': 'buy' if entry_conditions['direction'] == 'long' else 'sell',
                        'price': current['close'],
                        'size': size,
                        'type': 'market',
                        'reason': entry_conditions['reason'],
                        'parameters': {
                            'signal_strength': signal_strength,
                            'rsi': current['rsi'],
                            'macd_hist': current['macd_hist'],
                            'adx': current['adx'],
                            'regime': current['regime']
                        }
                    }
                    signals.append(signal)
                    
                    # Update tracking variables for trade frequency
                    self.last_signal_time = current.name
                    self.trades_today += 1
                
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
    
    def _reset_daily_counters(self, data: pd.DataFrame):
        """Reset trade counters when a new day begins"""
        if len(data) == 0:
            return
            
        # Get date of last candle
        last_date = data.index[-1].date()
        
        # Reset counter if it's a new day
        if self.current_day != last_date:
            self.current_day = last_date
            self.trades_today = 0
    
    def _trade_limit_reached(self, current: pd.Series) -> bool:
        """Check if we've reached trading frequency limits"""
        # Check daily trade limit
        max_trades_per_day = self.params.get('max_trades_per_day', 5)
        if self.trades_today >= max_trades_per_day:
            return True
            
        # Check minimum time between trades
        if self.last_signal_time is not None:
            min_hours = self.params.get('min_hours_between_trades', 4)
            time_since_last = (current.name - self.last_signal_time).total_seconds() / 3600
            if time_since_last < min_hours:
                return True
                
        return False
    
    def _calculate_signal_strength(self, current: pd.Series, prev: pd.Series, prev2: pd.Series) -> float:
        """Calculate momentum signal strength with improved algorithm"""
        try:
            # RSI component - stronger at extremes
            rsi = current['rsi']
            if rsi < 30:
                rsi_strength = (30 - rsi) / 30  # Higher strength as RSI approaches 0
            elif rsi > 70:
                rsi_strength = (rsi - 70) / 30  # Higher strength as RSI approaches 100
            else:
                rsi_strength = 0.2  # Moderate strength in middle range
                
            # MACD component - consider both absolute value and direction
            macd_hist = current['macd_hist']
            macd_prev = prev['macd_hist']
            macd_direction = np.sign(macd_hist) == np.sign(macd_prev)  # Consistent direction
            macd_strength = min(abs(macd_hist) / current['close'] * 100, 1.0)  # Normalized
            
            if macd_direction:
                macd_strength *= 1.2  # Boost for consistent direction
            
            # ADX component - stronger trends get higher weight
            adx = current['adx']
            adx_strength = min((adx - 15) / 40, 1.0) if adx > 15 else 0
            
            # Volume component - higher volume gets more weight
            volume_strength = min(current['volume_ratio'] / 2, 1.0) if 'volume_ratio' in current else 0.5
            
            # Trend alignment component
            trend_aligned = (current['close'] > current['ema_50'] and macd_hist > 0) or \
                           (current['close'] < current['ema_50'] and macd_hist < 0)
            trend_factor = 1.2 if trend_aligned else 0.8
            
            # Combine components with weights
            strength = (
                0.3 * rsi_strength +
                0.3 * macd_strength +
                0.2 * adx_strength +
                0.2 * volume_strength
            ) * trend_factor
            
            return max(0.1, min(strength, 1.0))  # Ensure value between 0.1 and 1.0
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.3  # Default to moderate strength on error
    
    def _check_entry_conditions(self, current: pd.Series, prev: pd.Series) -> Dict:
        """Check entry conditions with stricter criteria"""
        try:
            # Initialize result
            result = {
                'should_enter': False,
                'direction': None,
                'reason': None
            }
            
            # Get parameters with defaults if not specified
            rsi_oversold = self.params.get('rsi_oversold', 25)  # More conservative than traditional 30
            rsi_overbought = self.params.get('rsi_overbought', 75)  # More conservative than traditional 70
            adx_threshold = self.params.get('adx_threshold', 30)  # Higher than traditional 25
            require_volume = self.params.get('require_volume_confirmation', True)
            
            # Define volume confirmation threshold
            volume_threshold = 1.2  # Volume should be 20% above average
            
            # Check long entry conditions
            long_conditions = [
                current['rsi'] < rsi_oversold,  # Oversold condition
                current['macd_hist'] > prev['macd_hist'],  # Rising momentum
                current['adx'] > adx_threshold,  # Strong trend
                current['slowk'] > current['slowd']  # Stochastic confirmation
            ]
            
            # Add volume confirmation if required
            if require_volume and 'volume_ratio' in current:
                long_conditions.append(current['volume_ratio'] > volume_threshold)
                
            # Add trend alignment if required
            if self.params.get('require_trend_alignment', True):
                long_conditions.append(current['close'] > current['ema_50'])
            
            # Check short entry conditions
            short_conditions = [
                current['rsi'] > rsi_overbought,  # Overbought condition
                current['macd_hist'] < prev['macd_hist'],  # Falling momentum
                current['adx'] > adx_threshold,  # Strong trend
                current['slowk'] < current['slowd']  # Stochastic confirmation
            ]
            
            # Add volume confirmation if required
            if require_volume and 'volume_ratio' in current:
                short_conditions.append(current['volume_ratio'] > volume_threshold)
                
            # Add trend alignment if required
            if self.params.get('require_trend_alignment', True):
                short_conditions.append(current['close'] < current['ema_50'])
            
            # Evaluate conditions
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
        """Check exit conditions with improved criteria"""
        try:
            # Initialize result
            result = {
                'should_exit': False,
                'reason': None
            }
            
            # Get parameters with defaults
            rsi_neutral_low = self.params.get('rsi_neutral_low', 45)  # Exit long positions as RSI moves to neutral
            rsi_neutral_high = self.params.get('rsi_neutral_high', 55)  # Exit short positions as RSI moves to neutral
            
            # Exit conditions for long positions
            long_exit_conditions = [
                # RSI no longer oversold and moving toward neutral
                current['rsi'] > rsi_neutral_low,
                # MACD momentum slowing
                current['macd_hist'] < prev['macd_hist'],
                # Stochastic K crossing below D
                current['slowk'] < current['slowd'],
                # Price crossing below short-term moving average
                current['close'] < current['ema_9']
            ]
            
            # Exit conditions for short positions
            short_exit_conditions = [
                # RSI no longer overbought and moving toward neutral
                current['rsi'] < rsi_neutral_high,
                # MACD momentum slowing
                current['macd_hist'] > prev['macd_hist'],
                # Stochastic K crossing above D
                current['slowk'] > current['slowd'],
                # Price crossing above short-term moving average
                current['close'] > current['ema_9']
            ]
            
            # Determine exit based on multiple confirmations (3+ conditions)
            # This avoids exiting too early on minor price movements
            if sum(long_exit_conditions) >= 3:
                result.update({
                    'should_exit': True,
                    'reason': 'momentum_reversal_long'
                })
            elif sum(short_exit_conditions) >= 3:
                result.update({
                    'should_exit': True,
                    'reason': 'momentum_reversal_short'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            return {'should_exit': False, 'reason': None}
    
    def _calculate_position_size(self, candle: pd.Series, signal_strength: float) -> float:
        """Calculate position size with improved risk management"""
        try:
            # Get current capital from params
            current_capital = self.params.get('current_capital', self.params.get('capital', 10000))
            
            # Get risk parameters with safer defaults
            risk_factor = self.params.get('risk_factor', 0.01)  # 1% default risk per trade
            max_pos_size_pct = self.params.get('max_position_size', 0.05)  # 5% max position
            min_pos_size_pct = self.params.get('min_position_size', 0.01)  # 1% min position
            
            # Calculate base position size based on risk
            base_size_usd = risk_factor * current_capital
            
            # Adjust for signal strength - stronger signals get closer to max size
            strength_adjusted_size = base_size_usd * (0.5 + signal_strength * 0.5)
            
            # Convert to quantity based on current price
            quantity = strength_adjusted_size / candle['close']
            
            # Adjust for volatility using ATR - reduce size in volatile markets
            if 'atr' in candle:
                vol_ratio = candle['atr'] / candle['close']
                vol_factor = max(0.5, 1.0 - vol_ratio * 10)  # Reduce size as volatility increases
                quantity *= vol_factor
            
            # Apply position size limits based on capital
            max_quantity = (max_pos_size_pct * current_capital) / candle['close']
            min_quantity = (min_pos_size_pct * current_capital) / candle['close']
            
            # Additional safety check - ensure we're not risking too much on one trade
            if 'atr' in candle:
                # Use ATR to estimate stop loss distance
                stop_distance = candle['atr'] * 1.5  # 1.5x ATR for stop loss
                max_risk_quantity = (current_capital * risk_factor) / stop_distance
                # Use the more conservative of the two limits
                max_quantity = min(max_quantity, max_risk_quantity)
            
            return max(min_quantity, min(quantity, max_quantity))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0  # Return 0 size on error as a safety measure
    
    def _detect_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regime (trending, ranging, volatile)
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with market regime labels
        """
        try:
            # Calculate metrics for regime detection
            if 'atr' in data.columns and 'close' in data.columns:
                volatility = data['atr'] / data['close']
            else:
                # Calculate simple volatility if ATR not available
                volatility = data['close'].rolling(14).std() / data['close']
                
            # Calculate trend direction and strength
            if 'ema_50' in data.columns:
                direction = (data['close'] - data['ema_50']) / data['ema_50']
            else:
                direction = data['close'].pct_change(20)  # 20-period return
            
            # Get ADX if available or default to 15
            adx = data['adx'] if 'adx' in data.columns else pd.Series(15, index=data.index)
            
            # Define regimes (default to ranging)
            regime = pd.Series('ranging', index=data.index)
            
            # Trending when ADX high and consistent direction
            trending_mask = (adx > 25) & (abs(direction) > 0.02)
            regime[trending_mask] = 'trending'
            
            # Volatile when large price swings
            if len(volatility) > 50:
                volatile_mask = volatility > volatility.rolling(50).mean() * 1.5
                regime[volatile_mask] = 'volatile'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            # Default to 'ranging' as the most conservative regime
            return pd.Series('ranging', index=data.index)