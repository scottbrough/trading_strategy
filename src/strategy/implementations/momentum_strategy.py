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
        # Store trade history for position sizing adaptation
        self.recent_trades = []  
        # Track highest price during trade for trailing stop
        self.highest_high = {}
    
    def generate_signals(self, data: pd.DataFrame, higher_tf_df: pd.DataFrame = None) -> List[Dict]:
        """
        Generate trading signals based on momentum indicators with improved quality
        
        Args:
            data: DataFrame with OHLCV and indicator data
            higher_tf_df: Optional higher timeframe data for confirmation
            
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
            
            # Add trailing high/low calculations
            self._add_trailing_price_data(data)
            
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
                
                # Skip if not in a favorable market regime
                if current['regime'] != 'trending' and self.params.get('require_trend_confirmation', True):
                    continue
                
                # Check higher timeframe alignment if provided
                if higher_tf_df is not None and not self._check_higher_timeframe_alignment(current, higher_tf_df):
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
                            'regime': current['regime'],
                            'confluence_score': entry_conditions.get('confluence_score', 0)
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
    
    def _add_trailing_price_data(self, data: pd.DataFrame) -> None:
        """Add trailing high/low calculations for each symbol"""
        # Calculate highest high over last 10 periods for trailing stop
        data['highest_high'] = data['high'].rolling(10).max()
        data['lowest_low'] = data['low'].rolling(10).min()
        # Track bars in trade for time-based exit
        data['bars_in_trade'] = 0  # This would be updated during backtest or live trading
    
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
        """Check entry conditions with stricter criteria and confluence scoring"""
        try:
            # Initialize result
            result = {
                'should_enter': False,
                'direction': None,
                'reason': None,
                'confluence_score': 0
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
                current['slowk'] > current['slowd'],  # Stochastic confirmation
                current['close'] > current['ema_9']  # Price above short-term MA
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
                current['slowk'] < current['slowd'],  # Stochastic confirmation
                current['close'] < current['ema_9']  # Price below short-term MA
            ]
            
            # Add volume confirmation if required
            if require_volume and 'volume_ratio' in current:
                short_conditions.append(current['volume_ratio'] > volume_threshold)
                
            # Add trend alignment if required
            if self.params.get('require_trend_alignment', True):
                short_conditions.append(current['close'] < current['ema_50'])
            
            # Calculate confluence scores
            long_confluence_score = sum(1 for cond in long_conditions if cond)
            short_confluence_score = sum(1 for cond in short_conditions if cond)
            
            # Require stronger confluence (at least 4 out of 5 conditions)
            required_confluence = 4  # Stricter requirement
            
            # Evaluate conditions with stricter requirements
            if long_confluence_score >= required_confluence:
                result.update({
                    'should_enter': True,
                    'direction': 'long',
                    'reason': f'momentum_long_{long_confluence_score}',
                    'confluence_score': long_confluence_score
                })
            elif short_confluence_score >= required_confluence:
                result.update({
                    'should_enter': True,
                    'direction': 'short',
                    'reason': f'momentum_short_{short_confluence_score}',
                    'confluence_score': short_confluence_score
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking entry conditions: {str(e)}")
            return {'should_enter': False, 'direction': None, 'reason': None}
    
    def _check_exit_conditions(self, current: pd.Series, prev: pd.Series) -> Dict:
        """Check exit conditions with improved criteria including trailing stops"""
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
            
            # Add trailing stop logic - exit when price falls below recent high
            if 'highest_high' in current:
                trail_percent = 0.02  # 2% trailing stop
                trailing_stop_level = current['highest_high'] * (1 - trail_percent)
                
                if current['close'] < trailing_stop_level:
                    result.update({
                        'should_exit': True,
                        'reason': 'trailing_stop_triggered'
                    })
            
            # Add time-based exit - exit if too many bars without reaching profit target
            if 'bars_in_trade' in current and current['bars_in_trade'] > 10:
                result.update({
                    'should_exit': True, 
                    'reason': 'time_exit'
                })
            
            # Add trailing stop logic
            if 'highest_high' in current and self._has_position('long'):
                trail_percent = 0.02  # 2% trailing stop
                trailing_stop_level = current['highest_high'] * (1 - trail_percent)
                if current['close'] < trailing_stop_level:
                    return {'should_exit': True, 'reason': 'trailing_stop'}
            
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
    
    def _check_higher_timeframe_alignment(self, current_df: pd.Series, higher_tf_df: pd.DataFrame) -> bool:
        """Check if higher timeframe trend aligns with entry direction"""
        try:
            if higher_tf_df is None or higher_tf_df.empty:
                return True  # If no higher timeframe data, consider aligned
                
            # Get last row of higher timeframe data
            higher_current = higher_tf_df.iloc[-1]
            
            # Check if EMAs are aligned in higher timeframe
            ema_aligned = higher_current['ema_9'] > higher_current['ema_21'] > higher_current['ema_50']
            
            # Check MACD momentum in higher timeframe
            macd_positive = higher_current['macd_hist'] > 0
            
            return ema_aligned and macd_positive  # For long entries
            
        except Exception as e:
            logger.error(f"Error checking higher timeframe alignment: {str(e)}")
            return True  # Default to allow trades if check fails
    
    def _calculate_position_size(self, candle: pd.Series, signal_strength: float) -> float:
        """Calculate position size with improved risk management and adaptation"""
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
            
            # Adjust for market regime
            if 'regime' in candle:
                regime_factors = {
                    'trending': 1.0,  # Full size in trending markets
                    'ranging': 0.6,   # Reduced size in ranging markets
                    'volatile': 0.4   # Significantly reduced in volatile markets
                }
                regime = candle['regime']
                regime_factor = regime_factors.get(regime, 0.5)
                quantity *= regime_factor
            
            # Adjust for volatility using ATR - reduce size in volatile markets
            if 'atr' in candle:
                vol_ratio = candle['atr'] / candle['close']
                vol_factor = max(0.5, 1.0 - vol_ratio * 10)  # Reduce size as volatility increases
                quantity *= vol_factor
            
            # Adjust for win rate (reduce after consecutive losses)
            if hasattr(self, 'recent_trades') and len(self.recent_trades) >= 3:
                recent_results = [t['pnl'] > 0 for t in self.recent_trades[-3:]]
                # If 2 or more recent losses, reduce size
                if sum(recent_results) < 2:
                    quantity *= 0.75
            
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
        Detect market regime (trending, ranging, volatile) with enhanced criteria
        
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
                
            # Calculate volatility regime transitions detection
            volatility_change = volatility.pct_change(5)
            vol_increasing = volatility_change > 0.1
            
            # Calculate trend direction and strength
            if 'ema_50' in data.columns:
                direction = (data['close'] - data['ema_50']) / data['ema_50']
            else:
                direction = data['close'].pct_change(20)  # 20-period return
            
            # Get ADX if available or default to 15
            adx = data['adx'] if 'adx' in data.columns else pd.Series(15, index=data.index)
            
            # Define regimes (default to ranging)
            regime = pd.Series('ranging', index=data.index)
            
            # Add choppiness index
            high_low_range = data['high'].rolling(14).max() - data['low'].rolling(14).min()
            close_change = abs(data['close'] - data['close'].shift(14))
            choppiness = 1 - (close_change / high_low_range)
            choppy_market = choppiness > 0.5
            
            # Only mark trending when volatility is stable
            trending_mask = (adx > 25) & (abs(direction) > 0.02) & ~vol_increasing & ~choppy_market
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
    
    def _has_position(self, side: str = None) -> bool:
        """Check if we have an open position of the specified side"""
        # Simply returns False for backtest purposes
        # The actual position tracking is handled by the backtest engine
        return False