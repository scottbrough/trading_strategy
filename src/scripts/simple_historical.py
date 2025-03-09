#!/usr/bin/env python
"""
Script to create a properly structured configuration file for momentum strategy.
"""

import yaml
import os
from pathlib import Path

# Configuration structure with all required parameters
momentum_config = {
    'strategy': {
        # Required parameters that were missing
        'risk_fraction': 0.02,
        'max_position_size': 0.2,
        'min_position_size': 0.01,
        'stop_loss': 0.05,
        'profit_target': 0.1,
        
        # Strategy-specific parameters
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'adx_threshold': 25,
        'risk_factor': 0.02,
        'capital': 10000,
        
        # Momentum indicators
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
}

# Create config directory if it doesn't exist
config_dir = Path('config')
config_dir.mkdir(exist_ok=True)

# Write configuration to file
config_path = config_dir / 'momentum_config.yaml'
with open(config_path, 'w') as f:
    yaml.dump(momentum_config, f, default_flow_style=False)

print(f"Created configuration file: {config_path}")
print("You can now run your backtest with:")
print("./run_backtest.sh --strategy momentum_strategy --config config/momentum_config.yaml")