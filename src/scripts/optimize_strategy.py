#!/usr/bin/env python
"""
Script to optimize trading strategy parameters using Optuna framework.
Simpler implementation that avoids complex market regime detection.
"""

import sys
from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import optuna  # Make sure this is installed

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.implementations.momentum_strategy import MomentumStrategy
from src.data.database import db
from src.core.logger import log_manager
from src.strategy.backtest import BacktestEngine

logger = log_manager.get_logger(__name__)

def main():
    """Run strategy optimization using Optuna"""
    # Define date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    logger.info(f"Loading data for optimization from {start_date} to {end_date}")
    
    # Load data
    data = {}
    for symbol in ['BTC/USD']:
        df = db.get_ohlcv(symbol, '1h', start_date, end_date)
        if df.empty:
            logger.error(f"No data found for {symbol} in specified date range")
            return False
        data[symbol] = df
        logger.info(f"Loaded {len(df)} data points for {symbol}")
    
    # Base configuration - these parameters won't be optimized
    base_config = {
        'strategy': {
            'risk_fraction': 0.02,
            'max_position_size': 0.2,
            'min_position_size': 0.01,
            'risk_factor': 0.02,
        }
    }
    
    # Create backtest engine
    backtest_engine = BacktestEngine()
    
    # Define Optuna objective function
    def objective(trial):
        # Generate parameters using Optuna
        params = {
            'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 40),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 60, 80),
            'adx_threshold': trial.suggest_int('adx_threshold', 15, 35),
            'require_volume_confirmation': trial.suggest_categorical('require_volume_confirmation', [True, False]),
            'require_trend_alignment': trial.suggest_categorical('require_trend_alignment', [True, False]),
            'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.07),
            'profit_target': trial.suggest_float('profit_target', 0.02, 0.15),
            # Extra parameters to increase trade frequency
            'max_trades_per_day': trial.suggest_int('max_trades_per_day', 3, 10),
            'min_hours_between_trades': trial.suggest_int('min_hours_between_trades', 1, 6)
        }
        
        # Create full configuration
        config = base_config.copy()
        config['strategy'].update(params)
        
        # Create strategy instance
        strategy = MomentumStrategy(config)
        
        # Run backtest
        results = backtest_engine.run_backtest(
            strategy,
            data,
            initial_capital=10000,
            transaction_costs=True
        )
        
        # Calculate fitness (can be customized based on what you value most)
        metrics = results['metrics']
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        trade_count = metrics.get('total_trades', 0)
        
        # Higher returns, higher win rate, lower drawdown, and minimum number of trades
        # Special focus on increasing trade count while maintaining positive return
        score = (
            total_return * 0.4 +
            win_rate * 0.2 -
            max_drawdown * 0.2 +
            min(1.0, trade_count / 20) * 0.2  # Encourage at least 20 trades
        )
        
        # Heavily penalize strategies with too few trades
        if trade_count < 10:
            score *= 0.5
        
        # Store additional info for analysis
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('trade_count', trade_count)
        trial.set_user_attr('total_return', total_return)
        trial.set_user_attr('win_rate', win_rate)
        trial.set_user_attr('max_drawdown', max_drawdown)
        
        return score
    
    # Create study and optimize
    logger.info("Starting Optuna optimization (this may take a while)")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    # Log results
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_trial.value}")
    logger.info(f"Trade count: {best_trial.user_attrs['trade_count']}")
    logger.info(f"Total return: {best_trial.user_attrs['total_return']:.2%}")
    logger.info(f"Win rate: {best_trial.user_attrs['win_rate']:.2%}")
    logger.info(f"Max drawdown: {best_trial.user_attrs['max_drawdown']:.2%}")
    
    # Create final config
    final_config = base_config.copy()
    final_config['strategy'].update(best_params)
    
    # Save as YAML
    output_dir = Path("config")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "optimized_momentum.yaml", "w") as f:
        yaml.dump(final_config, f)
    
    logger.info(f"Optimized config saved to config/optimized_momentum.yaml")
    
    # Also save detailed results as JSON for further analysis
    results = {
        'best_parameters': best_params,
        'best_metrics': {
            'total_return': best_trial.user_attrs['total_return'],
            'win_rate': best_trial.user_attrs['win_rate'],
            'max_drawdown': best_trial.user_attrs['max_drawdown'],
            'trade_count': best_trial.user_attrs['trade_count'],
        },
        'all_trials': [
            {
                'params': trial.params,
                'value': trial.value,
                'metrics': {
                    'total_return': trial.user_attrs.get('total_return', 0),
                    'win_rate': trial.user_attrs.get('win_rate', 0),
                    'max_drawdown': trial.user_attrs.get('max_drawdown', 0),
                    'trade_count': trial.user_attrs.get('trade_count', 0),
                }
            }
            for trial in study.trials
        ]
    }
    
    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Detailed results saved to config/optimization_results.json")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Optimization completed successfully!")
    else:
        logger.error("Optimization failed!")