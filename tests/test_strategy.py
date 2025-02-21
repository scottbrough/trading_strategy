"""
Test suite for trading strategies and system components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from src.strategy.implementations.momentum_strategy import MomentumStrategy
from src.strategy.implementations.trend_strategy import TrendStrategy
from src.strategy.implementations.mean_reversion_strategy import MeanReversionStrategy
from src.data.processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    data = {
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(101, 2, 100),
        'low': np.random.normal(99, 2, 100),
        'close': np.random.normal(100, 2, 100),
        'volume': np.random.normal(1000, 100, 100)
    }
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.fixture
def momentum_strategy():
    """Create momentum strategy instance"""
    config = {
        'strategy': {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'adx_threshold': 25,
            'risk_factor': 0.02,
            'max_position_size': 0.1,
            'min_position_size': 0.01
        }
    }
    return MomentumStrategy(config)

@pytest.fixture
def trend_strategy():
    """Create trend strategy instance"""
    config = {
        'strategy': {
            'trend_threshold': 0.3,
            'quality_threshold': 0.6,
            'adx_threshold': 25,
            'risk_factor': 0.02,
            'max_position_size': 0.1,
            'min_position_size': 0.01
        }
    }
    return TrendStrategy(config)

@pytest.fixture
def mean_reversion_strategy():
    """Create mean reversion strategy instance"""
    config = {
        'strategy': {
            'zscore_threshold': 2.0,
            'probability_threshold': 0.7,
            'volume_threshold': 1.2,
            'risk_factor': 0.02,
            'max_position_size': 0.1,
            'min_position_size': 0.01
        }
    }
    return MeanReversionStrategy(config)

def test_momentum_strategy_signals(momentum_strategy, sample_data):
    """Test momentum strategy signal generation"""
    processor = DataProcessor()
    data = processor.process_ohlcv(sample_data, '1h')
    
    signals = momentum_strategy.generate_signals(data)
    
    assert isinstance(signals, list)
    if signals:
        assert all(isinstance(signal, dict) for signal in signals)
        assert all('action' in signal for signal in signals)
        assert all('timestamp' in signal for signal in signals)
        assert all('price' in signal for signal in signals)

def test_trend_strategy_signals(trend_strategy, sample_data):
    """Test trend strategy signal generation"""
    processor = DataProcessor()
    data = processor.process_ohlcv(sample_data, '1h')
    
    signals = trend_strategy.generate_signals(data)
    
    assert isinstance(signals, list)
    if signals:
        assert all(isinstance(signal, dict) for signal in signals)
        assert all('action' in signal for signal in signals)
        assert all('timestamp' in signal for signal in signals)
        assert all('price' in signal for signal in signals)

def test_mean_reversion_signals(mean_reversion_strategy, sample_data):
    """Test mean reversion strategy signal generation"""
    processor = DataProcessor()
    data = processor.process_ohlcv(sample_data, '1h')
    
    signals = mean_reversion_strategy.generate_signals(data)
    
    assert isinstance(signals, list)
    if signals:
        assert all(isinstance(signal, dict) for signal in signals)
        assert all('action' in signal for signal in signals)
        assert all('timestamp' in signal for signal in signals)
        assert all('price' in signal for signal in signals)

def test_position_sizing(momentum_strategy, sample_data):
    """Test position sizing calculations"""
    signal = {
        'action': 'buy',
        'price': 100.0,
        'signal_strength': 0.8
    }
    
    size = momentum_strategy._calculate_position_size(sample_data.iloc[-1], signal['signal_strength'])
    
    assert isinstance(size, float)
    assert size > 0
    assert size <= momentum_strategy.params['max_position_size']
    assert size >= momentum_strategy.params['min_position_size']

def test_risk_management(momentum_strategy):
    """Test risk management rules"""
    position = {
        'symbol': 'BTC/USD',
        'side': 'buy',
        'entry_price': 100.0,
        'size': 0.1,
        'timestamp': datetime.now()
    }
    
    # Test position limits
    assert momentum_strategy._check_position_limits(position)
    
    # Test risk per trade
    risk = momentum_strategy._calculate_position_risk(position)
    assert risk <= momentum_strategy.params['risk_factor']

def test_data_processing(sample_data):
    """Test data processing functionality"""
    processor = DataProcessor()
    processed_data = processor.process_ohlcv(sample_data, '1h')
    
    required_indicators = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_middle'
    ]
    
    assert all(indicator in processed_data.columns for indicator in required_indicators)
    assert not processed_data.isnull().any().any()

def test_strategy_optimization(momentum_strategy, sample_data):
    """Test strategy optimization"""
    processor = DataProcessor()
    data = processor.process_ohlcv(sample_data, '1h')
    
    # Test parameter optimization
    optimized_params = momentum_strategy.optimize_parameters(data)
    
    assert isinstance(optimized_params, dict)
    assert 'rsi_oversold' in optimized_params
    assert 'rsi_overbought' in optimized_params
    assert 'adx_threshold' in optimized_params

def test_performance_metrics(momentum_strategy, sample_data):
    """Test performance metrics calculation"""
    trades = [
        {'entry_price': 100, 'exit_price': 105, 'size': 1.0, 'side': 'buy'},
        {'entry_price': 105, 'exit_price': 103, 'size': 1.0, 'side': 'sell'}
    ]
    
    metrics = momentum_strategy.calculate_performance_metrics(trades)
    
    assert 'total_pnl' in metrics
    assert 'win_rate' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics