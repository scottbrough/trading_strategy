# tests/test_risk_management.py

"""
Tests for risk management system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategy.risk_management import RiskManager

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        """Set up risk manager for tests"""
        self.risk_config = {
            'max_drawdown': 0.15,
            'max_position_size': 0.2,
            'min_position_size': 0.01,
            'max_risk_per_trade': 0.02,
            'max_daily_loss': 0.05,
            'max_correlation': 0.7,
            'var_limit': -0.1
        }
        self.risk_manager = RiskManager(self.risk_config)
        
    def test_position_sizing(self):
        """Test position sizing calculations"""
        # Basic position sizing
        size = self.risk_manager.calculate_position_size(10000, 100, 0.1, {})
        self.assertGreater(size, 0)
        
        # Test position limits
        max_size = 10000 * self.risk_config['max_position_size'] / 100
        self.assertLessEqual(size, max_size)
        
        # Test with high volatility
        high_vol_size = self.risk_manager.calculate_position_size(10000, 100, 0.5, {})
        self.assertLess(high_vol_size, size)
        
    def test_risk_limits(self):
        """Test risk limit checks"""
        # Create a position
        position = {
            'symbol': 'BTC/USD',
            'side': 'long',
            'entry_price': 50000,
            'size': 0.1
        }
        
        # Test with normal conditions
        self.assertTrue(self.risk_manager._check_position_limits(position))
        
        # Test with too large position
        large_position = position.copy()
        large_position['size'] = 1.0
        self.assertFalse(self.risk_manager._check_position_limits(large_position))