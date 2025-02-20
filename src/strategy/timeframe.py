"""
Multi-timeframe analysis system implementing hierarchical market analysis
and timeframe alignment strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from src.core.config import config
from src.core.logger import log_manager
from src.data.database import db_manager

logger = log_manager.get_logger(__name__)

class TimeframeAnalyzer:
    def __init__(self):
        self.timeframes = config.get_timeframes()
        self.timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
    def align_timeframes(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align data from different timeframes to ensure consistency"""
        try:
            aligned_data = {}
            base_timeframe = min(self.timeframes, key=lambda x: self.timeframe_minutes[x])
            base_df = data[base_timeframe]
            
            for timeframe in self.timeframes:
                if timeframe == base_timeframe:
                    aligned_data[timeframe] = base_df
                    continue
                
                # Resample to higher timeframe
                minutes = self.timeframe_minutes[timeframe]
                resampled = self._resample_ohlcv(base_df, minutes)
                
                # Align with actual data
                if timeframe in data:
                    actual_df = data[timeframe]
                    resampled = self._align_with_actual(resampled, actual_df)
                
                aligned_data[timeframe] = resampled
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error aligning timeframes: {str(e)}")
            raise

    def _resample_ohlcv(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe"""
        rule = f'{minutes}T'
        resampled = pd.DataFrame()
        
        resampled['open'] = df['open'].resample(rule).first()
        resampled['high'] = df['high'].resample(rule).max()
        resampled['low'] = df['low'].resample(rule).min()
        resampled['close'] = df['close'].resample(rule).last()
        resampled['volume'] = df['volume'].resample(rule).sum()
        
        return resampled

    def _align_with_actual(self, resampled: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        """Align resampled data with actual higher timeframe data"""
        common_index = actual.index.intersection(resampled.index)
        resampled.loc[common_index] = actual.loc[common_index]
        return resampled

    def analyze_timeframes(self, symbol: str, current_time: datetime) -> Dict[str, Dict]:
        """Perform multi-timeframe analysis"""
        try:
            analysis = {}
            
            for timeframe in self.timeframes:
                # Get data for each timeframe
                minutes = self.timeframe_minutes[timeframe]
                lookback = int(minutes * 100)  # Get enough bars for analysis
                start_time = current_time - timedelta(minutes=lookback)
                
                df = db_manager.get_ohlcv(symbol, timeframe, start_time, current_time)
                if df.empty:
                    continue
                
                # Perform analysis for this timeframe
                analysis[timeframe] = self._analyze_single_timeframe(df)
            
            # Combine analyses across timeframes
            return self._combine_timeframe_analyses(analysis)
            
        except Exception as e:
            logger.error(f"Error in timeframe analysis: {str(e)}")
            raise

    def _analyze_single_timeframe(self, df: pd.DataFrame) -> Dict:
        """Analyze a single timeframe"""
        analysis = {}
        
        # Trend Analysis
        analysis['trend'] = self._analyze_trend(df)
        
        # Support/Resistance
        analysis['levels'] = self._find_support_resistance(df)
        
        # Volatility
        analysis['volatility'] = self._analyze_volatility(df)
        
        # Volume Analysis
        analysis['volume'] = self._analyze_volume(df)
        
        return analysis

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze price trend"""
        try:
            # Calculate EMAs
            df['ema20'] = df['close'].ewm(span=20).mean()
            df['ema50'] = df['close'].ewm(span=50).mean()
            
            # Determine trend direction
            current_price = df['close'].iloc[-1]
            ema20 = df['ema20'].iloc[-1]
            ema50 = df['ema50'].iloc[-1]
            
            # Trend strength using ADX-like calculation
            price_changes = df['close'].pct_change()
            trend_strength = abs(price_changes.mean()) / price_changes.std() if len(price_changes) > 1 else 0
            
            return {
                'direction': 'uptrend' if ema20 > ema50 else 'downtrend',
                'strength': trend_strength,
                'price_location': {
                    'above_ema20': current_price > ema20,
                    'above_ema50': current_price > ema50
                }
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            raise

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        try:
            # Find swing highs and lows
            window = 5
            highs = df['high'].rolling(window=window, center=True).max()
            lows = df['low'].rolling(window=window, center=True).min()
            
            # Identify potential levels
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(df) - window):
                if highs.iloc[i] == df['high'].iloc[i]:
                    resistance_levels.append(df['high'].iloc[i])
                if lows.iloc[i] == df['low'].iloc[i]:
                    support_levels.append(df['low'].iloc[i])
            
            # Cluster nearby levels
            resistance_levels = self._cluster_levels(resistance_levels)
            support_levels = self._cluster_levels(support_levels)
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {str(e)}")
            raise

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.005) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
            
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return sorted(clusters)

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze price volatility"""
        try:
            # Calculate various volatility metrics
            returns = df['close'].pct_change()
            current_vol = returns.std()
            
            # True Range
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Volatility state
            hist_vol = returns.rolling(30).std()
            vol_percentile = returns.std().rank(pct=True)
            
            return {
                'current': current_vol,
                'atr': atr,
                'state': 'high' if vol_percentile > 0.7 else 'low' if vol_percentile < 0.3 else 'normal',
                'percentile': vol_percentile
            }
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            raise

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze trading volume"""
        try:
            volume = df['volume']
            avg_volume = volume.rolling(20).mean()
            
            return {
                'current': volume.iloc[-1],
                'average': avg_volume.iloc[-1],
                'ratio': volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] != 0 else 0,
                'trend': 'increasing' if volume.iloc[-1] > avg_volume.iloc[-1] else 'decreasing'
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            raise

    def _combine_timeframe_analyses(self, analyses: Dict[str, Dict]) -> Dict:
        """Combine analyses from different timeframes"""
        try:
            combined = {
                'primary_trend': self._determine_primary_trend(analyses),
                'key_levels': self._combine_levels(analyses),
                'volatility_state': self._combine_volatility(analyses),
                'volume_profile': self._combine_volume(analyses)
            }
            
            # Add confluence analysis
            combined['confluence'] = self._analyze_confluence(combined)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining timeframe analyses: {str(e)}")
            raise

    def _determine_primary_trend(self, analyses: Dict[str, Dict]) -> str:
        """Determine the primary trend across timeframes"""
        trends = {}
        weights = {
            '1d': 0.4,
            '4h': 0.3,
            '1h': 0.2,
            '15m': 0.1
        }
        
        for timeframe, analysis in analyses.items():
            if timeframe in weights:
                trend_data = analysis.get('trend', {})
                direction = trend_data.get('direction', 'neutral')
                strength = trend_data.get('strength', 0)
                trends[timeframe] = (direction, strength, weights[timeframe])
        
        # Calculate weighted trend
        up_weight = sum(w for d, s, w in trends.values() if d == 'uptrend')
        down_weight = sum(w for d, s, w in trends.values() if d == 'downtrend')
        
        return 'uptrend' if up_weight > down_weight else 'downtrend'

    def _combine_levels(self, analyses: Dict[str, Dict]) -> Dict:
        """Combine support/resistance levels across timeframes"""
        all_levels = {
            'resistance': [],
            'support': []
        }
        
        for analysis in analyses.values():
            levels = analysis.get('levels', {})
            all_levels['resistance'].extend(levels.get('resistance', []))
            all_levels['support'].extend(levels.get('support', []))
        
        # Cluster combined levels
        return {
            'resistance': self._cluster_levels(all_levels['resistance']),
            'support': self._cluster_levels(all_levels['support'])
        }

    def _analyze_confluence(self, combined: Dict) -> Dict:
        """Analyze confluence of signals across timeframes"""
        confluence = {
            'strength': 0,
            'signals': []
        }
        
        # Check trend alignment
        if combined['primary_trend'] == 'uptrend':
            confluence['signals'].append('trend_aligned_up')
        elif combined['primary_trend'] == 'downtrend':
            confluence['signals'].append('trend_aligned_down')
        
        # Check level confluence
        levels = combined['key_levels']
        if levels['resistance'] and levels['support']:
            confluence['signals'].append('level_confluence')
        
        # Check volume confirmation
        if combined['volume_profile'].get('trend') == 'increasing':
            confluence['signals'].append('volume_confirmed')
        
        # Calculate overall confluence strength
        confluence['strength'] = len(confluence['signals']) / 3  # Normalize to 0-1
        
        return confluence

# Example usage
if __name__ == "__main__":
    analyzer = TimeframeAnalyzer()
    
    # Get current time
    current_time = datetime.utcnow()
    
    # Analyze BTC/USD across timeframes
    analysis = analyzer.analyze_timeframes("BTC/USD", current_time)
    print(json.dumps(analysis, indent=2))