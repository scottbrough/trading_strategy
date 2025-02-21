"""
Advanced mean-reversion strategy optimized for quick profitable trades.
Uses statistical analysis and multiple timeframe confirmation.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from ..base import BaseStrategy
from ..indicators import TechnicalIndicators
from ...core.logger import log_manager

logger = log_manager.get_logger(__name__)

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.indicators = TechnicalIndicators()
        self.required_indicators = [
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
            'rsi', 'atr', 'volume_ratio', 'macd_hist',
            'slowk', 'slowd'
        ]
        self.lookback_periods = config.get('lookback_periods', 20)
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Generate mean reversion signals"""
        try:
            signals = []
            
            # Process indicators if needed
            if not all(ind in data.columns for ind in self.required_indicators):
                data = self.indicators.calculate_all(data)
            
            # Add mean reversion metrics
            data['zscore'] = self._calculate_zscore(data['close'])
            data['mean_dev'] = self._calculate_mean_deviation(data)
            data['reversion_probability'] = self._calculate_reversion_probability(data)
            
            # Generate signals for each candle
            for i in range(2, len(data)):
                current = data.iloc[i]
                prev = data.iloc[i-1]
                
                # Calculate reversion conditions
                reversion_conditions = self._analyze_reversion_conditions(current, prev)
                
                if reversion_conditions['valid_signal']:
                    signal_strength = self._calculate_signal_strength(
                        current, reversion_conditions
                    )
                    
                    # Entry signals
                    if reversion_conditions['direction'] == 'long' and not self._has_position('long'):
                        signals.append(self._create_long_signal(
                            current, signal_strength, reversion_conditions
                        ))
                    elif reversion_conditions['direction'] == 'short' and not self._has_position('short'):
                        signals.append(self._create_short_signal(
                            current, signal_strength, reversion_conditions
                        ))
                    
                    # Exit signals
                    if self._check_exit_conditions(current, reversion_conditions):
                        signals.append({
                            'timestamp': current.name,
                            'action': 'exit',
                            'price': current['close'],
                            'type': 'market',
                            'reason': 'mean_reversion_complete'
                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {str(e)}")
            return []
    
    def _calculate_zscore(self, price: pd.Series) -> pd.Series:
        """Calculate z-score for price series"""
        try:
            rolling_mean = price.rolling(window=self.lookback_periods).mean()
            rolling_std = price.rolling(window=self.lookback_periods).std()
            return (price - rolling_mean) / rolling_std
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {str(e)}")
            return pd.Series(0, index=price.index)
    
    def _calculate_mean_deviation(self, data: pd.DataFrame) -> pd.Series:
        """Calculate mean deviation with volume weighting"""
        try:
            # Calculate volume-weighted mean
            vwap = (data['close'] * data['volume']).rolling(self.lookback_periods).sum() / \
                   data['volume'].rolling(self.lookback_periods).sum()
            
            # Calculate deviation
            deviation = (data['close'] - vwap) / data['atr']
            
            return deviation
            
        except Exception as e:
            logger.error(f"Error calculating mean deviation: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _calculate_reversion_probability(self, data: pd.DataFrame) -> pd.Series:
        """Calculate probability of mean reversion"""
        try:
            # Calculate historical probability of reversion based on z-score
            zscore = data['zscore']
            returns = data['close'].pct_change()
            
            # Look ahead 5 periods for reversion success rate
            future_returns = returns.shift(-5)
            
            # Calculate success rate for different z-score ranges
            high_zscore = zscore > 2
            low_zscore = zscore < -2
            
            # Calculate probabilities
            prob_reversion_high = (future_returns[high_zscore] < 0).mean() if high_zscore.any() else 0
            prob_reversion_low = (future_returns[low_zscore] > 0).mean() if low_zscore.any() else 0
            
            # Combine with volume and RSI confirmation
            volume_confirm = data['volume_ratio'] > 1.2
            rsi_confirm = (data['rsi'] < 30) | (data['rsi'] > 70)
            
            # Calculate final probability
            probability = pd.Series(index=data.index)
            probability[high_zscore] = prob_reversion_high
            probability[low_zscore] = prob_reversion_low
            probability[~(high_zscore | low_zscore)] = 0.5
            
            # Adjust probabilities with confirmations
            probability = probability * np.where(volume_confirm, 1.2, 1.0)
            probability = probability * np.where(rsi_confirm, 1.2, 1.0)
            
            return probability.clip(0, 1)
            
        except Exception as e:
            logger.error(f"Error calculating reversion probability: {str(e)}")
            return pd.Series(0, index=data.index)