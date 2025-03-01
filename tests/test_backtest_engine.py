"""
Tests for backtesting engine.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategy.backtest import BacktestEngine
from src.strategy.base import BaseStrategy

class MockStrategy(BaseStrategy):
    """Mock strategy for testing"""
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'strategy': {
                    'risk_fraction': 0.02,
                    'max_position_size': 0.2,
                    'stop_loss': 0.05,
                    'profit_target': 0.1
                }
            }
        super().__init__(config)
        
    def generate_signals(self, data):
        """Generate mock signals"""
        signals = []
        
        # Simple strategy: buy at 10th bar, sell at 20th
        if len(data) >= 10:
            signals.append({
                'timestamp': data.index[10],
                'action': 'buy',
                'price': data['close'].iloc[10],
                'size': 1.0
            })
            
        if len(data) >= 20:
            signals.append({
                'timestamp': data.index[20],
                'action': 'exit',
                'price': data['close'].iloc[20]
            })
            
        return signals
    
    def _validate_config(self):
        """Override validation to prevent errors during testing"""
        pass

class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        """Set up test data and engine"""
        # Create sample OHLCV data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='1d')
        data = {
            'open': np.random.normal(100, 5, 30),
            'high': np.random.normal(105, 5, 30),
            'low': np.random.normal(95, 5, 30),
            'close': np.random.normal(100, 5, 30),
            'volume': np.random.normal(1000, 200, 30)
        }
        self.test_df = pd.DataFrame(data, index=dates)
        self.engine = BacktestEngine()
        self.strategy = MockStrategy()
        
    def test_run_backtest(self):
        """Test running a backtest"""
        # Run backtest
        results = self.engine.run_backtest(
            self.strategy,
            {'TEST/USD': self.test_df},
            initial_capital=10000,
            transaction_costs=True
        )
        
        # Verify results
        self.assertIn('metrics', results)
        self.assertIn('equity_curve', results)
        self.assertIn('trades', results)
        
        # Should have executed a trade
        self.assertEqual(len(results['trades']), 1)
        
        # Check metrics
        metrics = results['metrics']
        self.assertIn('final_equity', metrics)
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)