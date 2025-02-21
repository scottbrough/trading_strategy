"""
Advanced trend-following strategy optimized for maximum profit.
Uses multiple timeframe analysis and dynamic trend detection.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..base import BaseStrategy
from ..indicators import TechnicalIndicators
from ...core.logger import log_manager

logger = log_manager.get_logger(__name__)

class TrendStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.indicators = TechnicalIndicators()
        self.required_indicators = [
            'ema_9', 'ema_21', 'ema_50', 'ema_200',
            'adx', 'atr', 'rsi', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_width',
            'volume_ratio', 'trend_strength'
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals based on trend analysis"""
        try:
            signals = []
            
            # Process indicators if needed
            if not all(ind in data.columns for ind in self.required_indicators):
                data = self.indicators.calculate_all(data)
            
            # Add trend strength calculation
            data['trend_score'] = self._calculate_trend_score(data)
            data['trend_quality'] = self._calculate_trend_quality(data)
            
            # Generate signals for each candle
            for i in range(2, len(data)):
                current = data.iloc[i]
                prev = data.iloc[i-1]
                prev2 = data.iloc[i-2]
                
                # Calculate trend conditions
                trend_conditions = self._analyze_trend_conditions(current, prev, prev2)
                
                if trend_conditions['valid_trend']:
                    signal_strength = self._calculate_signal_strength(
                        current, trend_conditions
                    )
                    
                    # Entry signals
                    if trend_conditions['trend_direction'] == 'long' and not self._has_position('long'):
                        signals.append(self._create_long_signal(
                            current, signal_strength, trend_conditions
                        ))
                    elif trend_conditions['trend_direction'] == 'short' and not self._has_position('short'):
                        signals.append(self._create_short_signal(
                            current, signal_strength, trend_conditions
                        ))
                    
                    # Exit signals
                    if self._check_exit_conditions(current, trend_conditions):
                        signals.append({
                            'timestamp': current.name,
                            'action': 'exit',
                            'price': current['close'],
                            'type': 'market',
                            'reason': 'trend_reversal'
                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trend signals: {str(e)}")
            return []
    
    def _calculate_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive trend score"""
        try:
            # EMA alignment score
            ema_score = np.where(
                (data['ema_9'] > data['ema_21']) &
                (data['ema_21'] > data['ema_50']) &
                (data['ema_50'] > data['ema_200']),
                1,
                np.where(
                    (data['ema_9'] < data['ema_21']) &
                    (data['ema_21'] < data['ema_50']) &
                    (data['ema_50'] < data['ema_200']),
                    -1,
                    0
                )
            )
            
            # ADX weight
            adx_weight = data['adx'] / 100.0
            
            # MACD trend
            macd_score = np.sign(data['macd_hist'])
            
            # Volume confirmation
            volume_score = np.where(data['volume_ratio'] > 1.5, 1, 0)
            
            # Combine scores
            trend_score = (
                0.4 * ema_score +
                0.3 * adx_weight * np.sign(ema_score) +
                0.2 * macd_score +
                0.1 * volume_score
            )
            
            return trend_score
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _calculate_trend_quality(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend quality metric"""
        try:
            # Volatility component
            volatility = data['atr'] / data['close']
            vol_score = 1 - (volatility / volatility.rolling(20).max())
            
            # Momentum consistency
            mom_score = abs(data['trend_score'].rolling(10).mean())
            
            # Price structure
            structure_score = 1 - data['bb_width'] / data['bb_width'].rolling(20).max()
            
            # Combine quality metrics
            quality = (
                0.4 * vol_score +
                0.4 * mom_score +
                0.2 * structure_score
            )
            
            return quality.clip(0, 1)
            
        except Exception as e:
            logger.error(f"Error calculating trend quality: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _analyze_trend_conditions(self, current: pd.Series, prev: pd.Series, prev2: pd.Series) -> Dict:
        """Analyze trend conditions"""
        try:
            trend_direction = np.sign(current['trend_score'])
            trend_quality = current['trend_quality']
            
            # Minimum trend requirements
            valid_trend = (
                abs(current['trend_score']) > self.params.get('trend_threshold', 0.3) and
                current['trend_quality'] > self.params.get('quality_threshold', 0.6) and
                current['adx'] > self.params.get('adx_threshold', 25)
            )
            
            return {
                'valid_trend': valid_trend,
                'trend_direction': 'long' if trend_direction > 0 else 'short',
                'trend_quality': trend_quality,
                'trend_strength': abs(current['trend_score']),
                'momentum': current['macd_hist'] / abs(current['macd_hist']).rolling(20).mean().iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend conditions: {str(e)}")
            return {
                'valid_trend': False,
                'trend_direction': None,
                'trend_quality': 0,
                'trend_strength': 0,
                'momentum': 0
            }
    
    def _calculate_signal_strength(self, current: pd.Series, trend_conditions: Dict) -> float:
        """Calculate signal strength for position sizing"""
        try:
            base_strength = trend_conditions['trend_strength']
            
            # Adjust for trend quality
            quality_factor = trend_conditions['trend_quality']
            
            # Adjust for momentum
            momentum_factor = min(abs(trend_conditions['momentum']), 2.0)
            
            # Adjust for volatility
            vol_factor = 1.0 - (current['atr'] / current['close'])
            
            signal_strength = (
                base_strength *
                quality_factor *
                momentum_factor *
                vol_factor
            )
            
            return min(signal_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.0
    
    def _create_long_signal(self, current: pd.Series, signal_strength: float,
                          trend_conditions: Dict) -> Dict:
        """Create long entry signal"""
        try:
            position_size = self._calculate_position_size(current, signal_strength)
            
            return {
                'timestamp': current.name,
                'action': 'buy',
                'price': current['close'],
                'size': position_size,
                'type': 'market',
                'reason': 'trend_following_long',
                'parameters': {
                    'trend_strength': trend_conditions['trend_strength'],
                    'trend_quality': trend_conditions['trend_quality'],
                    'signal_strength': signal_strength,
                    'adx': current['adx']
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating long signal: {str(e)}")
            return {}
    
    def _create_short_signal(self, current: pd.Series, signal_strength: float,
                           trend_conditions: Dict) -> Dict:
        """Create short entry signal"""
        try:
            position_size = self._calculate_position_size(current, signal_strength)
            
            return {
                'timestamp': current.name,
                'action': 'sell',
                'price': current['close'],
                'size': position_size,
                'type': 'market',
                'reason': 'trend_following_short',
                'parameters': {
                    'trend_strength': trend_conditions['trend_strength'],
                    'trend_quality': trend_conditions['trend_quality'],
                    'signal_strength': signal_strength,
                    'adx': current['adx']
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating short signal: {str(e)}")
            return {}
    
    def _check_exit_conditions(self, current: pd.Series, trend_conditions: Dict) -> bool:
        """Check exit conditions"""
        try:
            # Trend reversal
            trend_reversal = (
                abs(current['trend_score']) < self.params.get('trend_threshold', 0.3) or
                current['trend_quality'] < self.params.get('quality_threshold', 0.6)
            )
            
            # Momentum divergence
            momentum_reversal = (
                np.sign(current['macd_hist']) != 
                np.sign(current['trend_score'])
            )
            
            # Volatility spike
            volatility_exit = (
                current['atr'] > current['atr'].rolling(20).mean() * 2
            )
            
            return trend_reversal or momentum_reversal or volatility_exit
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            return False