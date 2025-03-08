"""
Backtest runner script for the trading system.
Allows running backtests from the command line with various parameters.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import importlib
import sys
import yaml
import numpy as np
from src.strategy.backtest import WalkForwardOptimizer

from ..core.logger import log_manager
from ..core.config import config
from ..strategy.backtest import BacktestEngine
from ..data.database import db

logger = log_manager.get_logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run trading strategy backtests')
    
    parser.add_argument('--strategy', type=str, required=True,
                      help='Strategy class to backtest (e.g., momentum_strategy.MomentumStrategy)')
    
    parser.add_argument('--symbols', type=str, nargs='+', required=True,
                      help='Symbols to backtest (e.g., BTC/USD ETH/USD)')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                      help='Timeframe to use (e.g., 1m, 5m, 15m, 1h, 4h, 1d)')
    
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                      help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--initial-capital', type=float, default=None,
                      help='Initial capital for backtest')
    
    parser.add_argument('--config', type=str, default=None,
                      help='Path to strategy configuration file')
    
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save backtest results')
    
    parser.add_argument('--no-transaction-costs', action='store_true',
                      help='Disable transaction costs in backtest')
    
    parser.add_argument('--plot', action='store_true',
                      help='Generate plots of backtest results')
    
    parser.add_argument('--optimize', action='store_true',
                        help='Run walk-forward optimization')
        
    parser.add_argument('--opt-folds', type=int, default=5,
                        help='Number of folds for walk-forward optimization')
        
    parser.add_argument('--opt-trials', type=int, default=20,
                        help='Number of parameter combinations to test per fold')
        
    parser.add_argument('--opt-params', type=str, default=None,
                        help='Path to parameter ranges JSON file')
        
    return parser.parse_args()


def load_parameter_ranges(param_file):
    """Load parameter ranges for optimization"""
    try:
        with open(param_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load parameter ranges: {str(e)}")
        return None


def load_strategy_config(config_path):
    """Load strategy configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load strategy config: {str(e)}")
        return {}


def load_strategy_class(strategy_path):
    """Dynamically load strategy class"""
    try:
        # Parse module and class name
        parts = strategy_path.split('.')
        class_name = parts[-1]
        module_path = '.'.join(parts[:-1])
        
        # Dynamically import module
        module = importlib.import_module(f"src.strategy.implementations.{module_path}")


        # Get class from module
        strategy_class = getattr(module, class_name)
        
        return strategy_class
    except Exception as e:
        logger.error(f"Failed to load strategy class: {str(e)}")
        sys.exit(1)


def load_data(symbols, timeframe, start_date, end_date):
    """Load data for backtest from database"""
    data_dict = {}
    
    for symbol in symbols:
        try:
            # Convert string dates to datetime if provided
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
            
            # Default date range if not provided
            if not start_dt:
                start_dt = datetime.now() - timedelta(days=365)  # 1 year
            if not end_dt:
                end_dt = datetime.now()
                
            # Get data from database
            df = db.get_ohlcv(symbol, timeframe, start_dt, end_dt)
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {timeframe}")
                continue
                
            data_dict[symbol] = df
            logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    if not data_dict:
        logger.error("No data loaded for any symbol")
        sys.exit(1)
                
    return data_dict


def run_backtest():
    """Run backtest with specified parameters"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load strategy class and configuration
    strategy_class = load_strategy_class(args.strategy)
    strategy_config = load_strategy_config(args.config) if args.config else {}
    
    # Load data
    data_dict = load_data(args.symbols, args.timeframe, args.start_date, args.end_date)
    
    # Check if we should run optimization
    if args.optimize:
        if args.opt_params:
            # Load parameter ranges
            parameter_ranges = load_parameter_ranges(args.opt_params)
            if not parameter_ranges:
                logger.error("Failed to load parameter ranges")
                sys.exit(1)
                
            # Create walk-forward optimizer
            optimizer = WalkForwardOptimizer(
                strategy_class,
                data_dict,
                parameter_ranges,
                strategy_config  # Use as base config
            )
            
            # Run optimization
            logger.info("Starting walk-forward optimization")
            results = optimizer.optimize(
                num_folds=args.opt_folds,
                num_trials=args.opt_trials
            )
            
            # Save optimization results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            opt_filename = f"optimization_{args.strategy.split('.')[-1]}_{timestamp}.json"
            
            with open(output_dir / opt_filename, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                sanitized_results = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.number) else o))
                json.dump(sanitized_results, f, indent=4)
                
            logger.info(f"Optimization results saved to {output_dir / opt_filename}")
            
            # Use optimized parameters for backtest
            strategy = strategy_class(results['best_parameters'])
        else:
            logger.error("Parameter ranges file required for optimization")
            sys.exit(1)
    else:
        # Instantiate strategy with normal config
        strategy = strategy_class(strategy_config)
    
    # Create backtest engine
    backtest_engine = BacktestEngine()
    
    # Run backtest
    logger.info(f"Starting backtest for {args.strategy} on {args.symbols}")
    results = backtest_engine.run_backtest(
        strategy,
        data_dict,
        initial_capital=args.initial_capital,
        transaction_costs=not args.no_transaction_costs
    )
    
    # Process results
    metrics = results['metrics']
    trades = results.get('trades', [])
    
    logger.info(f"Backtest completed with {len(trades)} trades")
    logger.info(f"Final equity: {metrics['final_equity']:.2f}")
    logger.info(f"Total return: {metrics['total_return']:.2%}")
    logger.info(f"Win rate: {metrics['win_rate']:.2%}")
    logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{args.strategy.split('.')[-1]}_{timestamp}"
    
    # Save JSON results
    backtest_engine.save_results(output_dir / f"{result_filename}.json")
    
    # Generate plots if requested
    if args.plot:
        backtest_engine.plot_results(output_dir / f"{result_filename}.png")
        
    logger.info(f"Backtest results saved to {output_dir}")
    
if __name__ == "__main__":
    run_backtest()