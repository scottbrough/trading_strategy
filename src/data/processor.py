"""
Data processing module for feature engineering and data transformation.
Handles data preprocessing, technical indicators, and multi-timeframe analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import talib
from datetime import datetime, timedelta
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor

from ..core.config import config
from ..core.logger import log_manager
from ..core.exceptions import DataError
from .database import db

logger = log_manager.get_logger(__name__)

class DataProcessor:
    def __init__(self):
        self.config = config
        self.timeframes = self.config.get_timeframes()
        self.symbols = self.config.get_symbols()
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def process_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Process OHLCV data with technical indicators and feature engineering
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe of the data
            
        Returns:
            DataFrame with additional features and indicators
        """
        try:
            if df.empty:
                raise DataError.DataValidationError("Empty DataFrame provided")
            
            # Make copy to avoid modifying original data
            df = df.copy()
            
            # Basic features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close']).diff()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price features
            df['price_ma_20'] = df['close'].rolling(window=20).mean()
            df['price_ma_50'] = df['close'].rolling(window=50).mean()
            df['price_std_20'] = df['close'].rolling(window=20).std()
            
            # Technical indicators
            self._add_technical_indicators(df)
            
            # Volatility features
            self._add_volatility_features(df)
            
            # Momentum features
            self._add_momentum_features(df)
            
            # Clean up
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {str(e)}")
            raise DataError.DataValidationError("Failed to process OHLCV data")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """Add technical indicators to the DataFrame"""
        try:
            # Trend indicators
            df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'],
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            
            # ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Ichimoku Cloud
            df['tenkan_sen'] = self._ichimoku_conversion(df, 9)
            df['kijun_sen'] = self._ichimoku_conversion(df, 26)
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise DataError.DataValidationError("Failed to add technical indicators")
    
    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        """Add volatility-based features"""
        try:
            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['atr_pct'] = df['atr'] / df['close']
            
            # Historical volatility
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['volatility_50'] = df['returns'].rolling(window=50).std() * np.sqrt(252)
            
            # Price channels
            df['high_channel'] = df['high'].rolling(20).max()
            df['low_channel'] = df['low'].rolling(20).min()
            df['channel_width'] = (df['high_channel'] - df['low_channel']) / df['close']
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {str(e)}")
            raise DataError.DataValidationError("Failed to add volatility features")
    
    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        """Add momentum-based features"""
        try:
            # ROC
            df['roc_5'] = talib.ROC(df['close'], timeperiod=5)
            df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
            df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
            
            # Momentum
            df['momentum'] = talib.MOM(df['close'], timeperiod=10)
            
            # Stochastic
            df['slowk'], df['slowd'] = talib.STOCH(
                df['high'],
                df['low'],
                df['close'],
                fastk_period=5,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            
        except Exception as e:
            logger.error(f"Error adding momentum features: {str(e)}")
            raise DataError.DataValidationError("Failed to add momentum features")
    
    @staticmethod
    def _ichimoku_conversion(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Ichimoku conversion line"""
        high_values = df['high'].rolling(window=period).max()
        low_values = df['low'].rolling(window=period).min()
        return (high_values + low_values) / 2
    
    def resample_timeframe(self, df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
        """Resample data to a different timeframe"""
        try:
            # Convert timeframe string to pandas offset
            tf_map = {
                '1m': '1Min', '5m': '5Min', '15m': '15Min',
                '1h': '1H', '4h': '4H', '1d': '1D'
            }
            
            if target_tf not in tf_map:
                raise DataError.DataValidationError(f"Invalid target timeframe: {target_tf}")
            
            # Resample OHLCV data
            resampled = df.resample(tf_map[target_tf]).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error resampling timeframe: {str(e)}")
            raise DataError.DataValidationError("Failed to resample timeframe")
    
    def get_multi_timeframe_data(self, symbol: str, base_timeframe: str = '1m',
                               start_time: datetime = None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        try:
            # Get base timeframe data
            if start_time is None:
                start_time = datetime.utcnow() - timedelta(days=30)
            
            base_data = db.get_ohlcv(symbol, base_timeframe, start_time, datetime.utcnow())
            
            # Process each timeframe
            mtf_data = {}
            for tf in self.timeframes:
                if tf == base_timeframe:
                    mtf_data[tf] = self.process_ohlcv(base_data, tf)
                else:
                    resampled = self.resample_timeframe(base_data, base_timeframe, tf)
                    mtf_data[tf] = self.process_ohlcv(resampled, tf)
            
            return mtf_data
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe data: {str(e)}")
            raise DataError.DataValidationError("Failed to get multi-timeframe data")
    
    @lru_cache(maxsize=100)
    def calculate_correlation(self, symbols: List[str], timeframe: str,
                            window: int = 50) -> pd.DataFrame:
        """Calculate correlation matrix between symbols"""
        try:
            returns_data = {}
            for symbol in symbols:
                df = db.get_ohlcv(symbol, timeframe,
                                datetime.utcnow() - timedelta(days=30),
                                datetime.utcnow())
                returns_data[symbol] = df['close'].pct_change()
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation = returns_df.rolling(window=window).corr()
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            raise DataError.DataValidationError("Failed to calculate correlation")
    
    def prepare_features(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                        feature_list: List[str] = None) -> pd.DataFrame:
        """Prepare feature matrix for model input"""
        try:
            if isinstance(data, dict):
                # Combine multi-timeframe data
                features = pd.DataFrame()
                for tf, df in data.items():
                    tf_features = df[feature_list] if feature_list else df
                    tf_features = tf_features.add_prefix(f'{tf}_')
                    features = pd.concat([features, tf_features], axis=1)
            else:
                features = data[feature_list] if feature_list else data
            
            # Scale features if needed
            # Add any additional feature preparation here
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise DataError.DataValidationError("Failed to prepare features")

# Example usage:
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Get and process data for a symbol
    symbol = "BTC/USD"
    mtf_data = processor.get_multi_timeframe_data(symbol)
    
    # Calculate correlations
    symbols = ["BTC/USD", "ETH/USD"]
    corr_matrix = processor.calculate_correlation(tuple(symbols), "1h")
    
    # Prepare features for modeling
    features = processor.prepare_features(mtf_data, ['close', 'volume', 'rsi', 'macd'])
    
    print("Data processing complete")