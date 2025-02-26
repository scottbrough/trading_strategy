"""
Tests for data processing functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create sample OHLCV data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='1d')
        data = {
            'open': np.random.normal(100, 5, 30),
            'high': np.random.normal(105, 5, 30),
            'low': np.random.normal(95, 5, 30),
            'close': np.random.normal(100, 5, 30),
            'volume': np.random.normal(1000, 200, 30)
        }
        
        # Ensure high > low
        for i in range(30):
            data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
            data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
        
        self.test_df = pd.DataFrame(data, index=dates)
        self.processor = DataProcessor()
        
    def test_process_ohlcv(self):
        """Test OHLCV data processing"""
        processed = self.processor.process_ohlcv(self.test_df, '1d')
        
        # Verify indicators were calculated
        self.assertIn('rsi', processed.columns)
        self.assertIn('macd', processed.columns)
        self.assertIn('bbands_upper', processed.columns)
        
        # Check that there are no NaN values in key columns
        self.assertFalse(processed['rsi'].iloc[-10:].isna().any())
        
    # tests/test_data_processor.py
    def test_resample_timeframe(self):
        """Test resampling to different timeframes"""
        # Create minute data
        dates = pd.date_range(start=datetime.now() - timedelta(hours=24), periods=24*60, freq='1min')
        data = {
            'open': np.random.normal(100, 5, 24*60),
            'high': np.random.normal(105, 5, 24*60),
            'low': np.random.normal(95, 5, 24*60),
            'close': np.random.normal(100, 5, 24*60),
            'volume': np.random.normal(100, 20, 24*60)
        }
        minute_df = pd.DataFrame(data, index=dates)
        
        # Resample to hourly
        hourly = self.processor.resample_timeframe(minute_df, '1m', '1h')
        
        # Verify
        self.assertEqual(len(hourly), 25)  # 24 hours
        self.assertEqual(hourly.index.freqstr, 'h')  # Changed from 'H' to 'h'