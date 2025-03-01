"""
Tests for strategy implementations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategy.implementations.momentum_strategy import MomentumStrategy
from src.strategy.implementations.trend_strategy import TrendStrategy
from src.strategy.implementations.mean_reversion_strategy import MeanReversionStrategy

class TestStrategies(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create sample OHLCV data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=50, freq='1d')
        
        # Create a more realistic price series with a trend
        base_price = 100
        trend = np.linspace(0, 10, 50)  # Upward trend
        noise = np.random.normal(0, 1, 50)  # Random noise
        price_series = base_price + trend + noise
        
        # Create OHLCV data
        data = {
            'open': price_series * 0.99,
            'high': price_series * 1.02,
            'low': price_series * 0.97,
            'close': price_series,
            'volume': np.random.normal(1000, 200, 50)
        }
        
        # Ensure high is always the highest and low is always the lowest
        for i in range(50):
            high_val = max(data['open'][i], data['close'][i], data['high'][i])
            low_val = min(data['open'][i], data['close'][i], data['low'][i])
            data['high'][i] = high_val * 1.01  # Ensure high is slightly higher
            data['low'][i] = low_val * 0.99   # Ensure low is slightly lower
        
        self.test_df = pd.DataFrame(data, index=dates)
        
        # Add some basic indicators for testing
        self.test_df['rsi'] = self._calculate_rsi(self.test_df['close'], 14)
        self.test_df['macd'], self.test_df['macd_signal'], self.test_df['macd_hist'] = self._calculate_macd(self.test_df['close'])
        self.test_df['ema_9'] = self.test_df['close'].ewm(span=9).mean()
        self.test_df['ema_21'] = self.test_df['close'].ewm(span=21).mean()
        self.test_df['ema_50'] = self.test_df['close'].ewm(span=50).mean()
        self.test_df['adx'] = 25  # Mock ADX value for testing
        self.test_df['atr'] = self._calculate_atr(self.test_df, 14)
        self.test_df['slowk'] = 50  # Mock stochastic values
        self.test_df['slowd'] = 50
        self.test_df['bb_upper'] = self.test_df['close'] + 2 * self.test_df['close'].rolling(20).std()
        self.test_df['bb_middle'] = self.test_df['close'].rolling(20).mean()
        self.test_df['bb_lower'] = self.test_df['close'] - 2 * self.test_df['close'].rolling(20).std()
        self.test_df['volume_ratio'] = 1.2  # Mock volume ratio
        
        # Strategy configurations - FIXED by putting parameters under 'strategy' key
        self.momentum_config = {
            'strategy': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'adx_threshold': 25,
                'risk_factor': 0.02,
                'capital': 10000,
                'max_position_size': 0.2,
                'min_position_size': 0.01,
                'stop_loss': 0.05,
                'profit_target': 0.1,
                'risk_fraction': 0.02
            }
        }
        
        self.trend_config = {
            'strategy': {
                'trend_threshold': 0.3,
                'quality_threshold': 0.6,
                'adx_threshold': 25,
                'risk_factor': 0.02,
                'capital': 10000,
                'max_position_size': 0.2,
                'min_position_size': 0.01,
                'stop_loss': 0.05,
                'profit_target': 0.1,
                'risk_fraction': 0.02
            }
        }
        
        self.reversion_config = {
            'strategy': {
                'lookback_periods': 20,
                'risk_factor': 0.02,
                'capital': 10000,
                'max_position_size': 0.2,
                'min_position_size': 0.01,
                'stop_loss': 0.05,
                'profit_target': 0.08,
                'risk_fraction': 0.02
            }
        }
    
    def _calculate_rsi(self, prices, period=14):
        """Simple RSI calculation for testing"""
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=period-1, adjust=False).mean()
        ema_down = down.ewm(com=period-1, adjust=False).mean()
        rs = ema_up / ema_down
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Simple MACD calculation for testing"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_atr(self, df, period=14):
        """Simple ATR calculation for testing"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def test_momentum_strategy(self):
        """Test momentum strategy signal generation"""
        # Initialize strategy
        strategy = MomentumStrategy(self.momentum_config)
        
        # Generate signals
        signals = strategy.generate_signals(self.test_df)
        
        # Verify signal structure and elements
        if signals:
            for signal in signals:
                self.assertIn('action', signal)
                self.assertIn('timestamp', signal)
                self.assertIn('price', signal)
                
                # Check valid action types
                self.assertIn(signal['action'], ['buy', 'sell', 'exit'])
                
                # Check price is valid
                self.assertGreater(signal['price'], 0)
                
                # For buy/sell, check size
                if signal['action'] in ['buy', 'sell']:
                    self.assertIn('size', signal)
                    self.assertGreater(signal['size'], 0)
        
    def test_trend_strategy(self):
        """Test trend strategy signal generation"""
        # Add trend indicators needed for the strategy
        self.test_df['trend_score'] = 0.5
        self.test_df['trend_quality'] = 0.7
        
        # Initialize strategy
        strategy = TrendStrategy(self.trend_config)
        
        # Generate signals
        signals = strategy.generate_signals(self.test_df)
        
        # Verify signal structure and elements
        if signals:
            for signal in signals:
                self.assertIn('action', signal)
                self.assertIn('timestamp', signal)
                self.assertIn('price', signal)
                
                # Check valid action types
                self.assertIn(signal['action'], ['buy', 'sell', 'exit'])
                
                # Check price is valid
                self.assertGreater(signal['price'], 0)
                
                # For buy/sell, check size
                if signal['action'] in ['buy', 'sell']:
                    self.assertIn('size', signal)
                    self.assertGreater(signal['size'], 0)
    
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy signal generation"""
        # Add reversion indicators needed for the strategy
        self.test_df['zscore'] = (self.test_df['close'] - self.test_df['close'].rolling(20).mean()) / self.test_df['close'].rolling(20).std()
        self.test_df['mean_dev'] = self.test_df['close'] - self.test_df['close'].rolling(20).mean()
        self.test_df['reversion_probability'] = 0.7
        
        try:
            # Initialize strategy
            strategy = MeanReversionStrategy(self.reversion_config)
            
            # Generate signals
            signals = strategy.generate_signals(self.test_df)
            
            # Verify signal structure and elements
            if signals:
                for signal in signals:
                    self.assertIn('action', signal)
                    self.assertIn('timestamp', signal)
                    self.assertIn('price', signal)
                    
                    # Check valid action types
                    self.assertIn(signal['action'], ['buy', 'sell', 'exit'])
                    
                    # Check price is valid
                    self.assertGreater(signal['price'], 0)
        except Exception as e:
            self.fail(f"Mean reversion strategy test failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()