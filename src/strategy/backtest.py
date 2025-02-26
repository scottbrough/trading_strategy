"""
Comprehensive backtesting engine for trading strategies.
Implements simulation of trading strategies with realistic execution modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from pathlib import Path
import copy

from ..core.logger import log_manager
from ..core.config import config
from ..core.exceptions import StrategyError
from ..data.database import db
from ..data.processor import DataProcessor

logger = log_manager.get_logger(__name__)

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, config_params: Dict[str, Any] = None):
        """Initialize backtest engine with configuration"""
        self.config = config_params or config.get_trading_params()
        self.data_processor = DataProcessor()
        self.results = {}
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, 
                   strategy: Any,
                   data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                   initial_capital: float = None,
                   start_date: datetime = None,
                   end_date: datetime = None,
                   transaction_costs: bool = True) -> Dict[str, Any]:
        """
        Run backtest for the given strategy and data
        
        Args:
            strategy: Strategy object to backtest
            data: DataFrame or dict of DataFrames with OHLCV data
            initial_capital: Starting capital (uses config if None)
            start_date: Start date for backtest (uses all data if None)
            end_date: End date for backtest (uses all data if None)
            transaction_costs: Whether to include transaction costs
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self._initialize_backtest(initial_capital)
            
            # Handle single DataFrame or dict of DataFrames
            if isinstance(data, pd.DataFrame):
                symbol = 'UNKNOWN'
                data_dict = {symbol: data}
            else:
                data_dict = data
                
            # Process data
            processed_data = self._prepare_data(data_dict, start_date, end_date)
            
            # Run strategy
            signals = self._generate_signals(strategy, processed_data)
            
            # Execute signals
            self._execute_signals(signals, processed_data, transaction_costs)
            
            # Calculate performance metrics
            metrics = self._calculate_metrics()
            
            # Store results
            self.results = {
                'metrics': metrics,
                'equity_curve': self.equity_curve,
                'trades': self.trades,
                'positions': self.positions
            }
            
            logger.info(f"Backtest completed: {len(self.trades)} trades, "
                       f"final equity: {metrics['final_equity']:.2f}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise StrategyError.BacktestError("Failed to run backtest", 
                                            details={'error': str(e)})
                
    def _initialize_backtest(self, initial_capital: float = None):
        """Initialize backtest state"""
        self.initial_capital = initial_capital or self.config.get('initial_balance', 10000)
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        
    def _prepare_data(self, data_dict: Dict[str, pd.DataFrame],
                    start_date: datetime = None,
                    end_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """Prepare data for backtesting"""
        processed_data = {}
        
        for symbol, df in data_dict.items():
            # Make a copy to avoid modifying original data
            df_copy = df.copy()
            
            # Filter by date if provided
            if start_date:
                df_copy = df_copy[df_copy.index >= start_date]
            if end_date:
                df_copy = df_copy[df_copy.index <= end_date]
                
            # Ensure OHLCV columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_copy.columns for col in required_cols):
                logger.warning(f"Missing required columns for {symbol}, skipping")
                continue
                
            # Add required indicators if not present
            processed_data[symbol] = self.data_processor.process_ohlcv(df_copy, '1d')
            
        if not processed_data:
            raise ValueError("No valid data after preparation")
            
        return processed_data
        
    def _generate_signals(self, strategy: Any,
                        data_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate trading signals from strategy"""
        all_signals = []
        
        for symbol, df in data_dict.items():
            try:
                # Generate signals for this symbol
                signals = strategy.generate_signals(df)
                
                # Add symbol to each signal
                for signal in signals:
                    if 'symbol' not in signal:
                        signal['symbol'] = symbol
                    all_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                continue
                
        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x['timestamp'])
        
        return all_signals
        
    def _execute_signals(self, signals: List[Dict[str, Any]],
                       data_dict: Dict[str, pd.DataFrame],
                       transaction_costs: bool = True):
        """Execute signals and track performance"""
        # Process each signal in order
        previous_date = None
        
        for signal in signals:
            try:
                # Get signal details
                symbol = signal.get('symbol', 'UNKNOWN')
                timestamp = signal['timestamp']
                action = signal['action']
                price = signal['price']
                
                # Check if we have a new date for equity tracking
                current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                if previous_date is None or current_date > previous_date:
                    self._update_equity_curve(timestamp)
                    previous_date = current_date
                
                # Execute signal based on action
                if action == 'buy':
                    self._open_position(symbol, timestamp, price, signal.get('size', 1.0), 'long')
                elif action == 'sell':
                    self._open_position(symbol, timestamp, price, signal.get('size', 1.0), 'short')
                elif action == 'exit':
                    self._close_positions(symbol, timestamp, price)
                    
                # Apply transaction costs if enabled
                if transaction_costs:
                    cost = price * signal.get('size', 1.0) * self.config.get('transaction_fee_rate', 0.001)
                    self.current_capital -= cost
                    
            except Exception as e:
                logger.error(f"Error executing signal: {str(e)}")
                continue
                
        # Close any remaining positions at the end of the backtest
        for position in self.positions[:]:
            symbol = position['symbol']
            if symbol in data_dict:
                last_price = data_dict[symbol]['close'].iloc[-1]
                last_date = data_dict[symbol].index[-1]
                self._close_position(position, last_date, last_price, 'backtest_end')
                
        # Final equity curve update
        if self.positions:
            last_date = max(data_dict[pos['symbol']].index[-1] for pos in self.positions
                          if pos['symbol'] in data_dict)
            self._update_equity_curve(last_date)
        
    def _open_position(self, symbol: str, timestamp: datetime, 
                     price: float, size: float, side: str):
        """Open a new position"""
        # Check if we have enough capital
        position_value = price * size
        if position_value > self.current_capital * self.config.get('max_position_size', 0.2):
            size = (self.current_capital * self.config.get('max_position_size', 0.2)) / price
            
        # Create position
        position = {
            'symbol': symbol,
            'entry_time': timestamp,
            'entry_price': price,
            'size': size,
            'side': side,
            'current_price': price,
            'unrealized_pnl': 0.0
        }
        
        self.positions.append(position)
        
        # Log position opening
        logger.debug(f"Opened {side} position in {symbol}: {size} @ {price}")
        
    def _close_position(self, position: Dict[str, Any], timestamp: datetime,
                      price: float, reason: str):
        """Close an existing position and record the trade"""
        # Calculate P&L
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        if side == 'long':
            pnl = (price - entry_price) * size
        else:  # short
            pnl = (entry_price - price) * size
            
        # Update capital
        self.current_capital += pnl
        
        # Record trade
        trade = {
            'symbol': position['symbol'],
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': price,
            'size': size,
            'side': side,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * size)) * 100,
            'reason': reason
        }
        
        self.trades.append(trade)
        
        # Remove position
        self.positions.remove(position)
        
        # Log trade
        logger.debug(f"Closed {side} position in {position['symbol']}: "
                    f"{size} @ {price}, PnL: {pnl:.2f}")
        
    def _close_positions(self, symbol: str, timestamp: datetime, price: float):
        """Close all positions for a given symbol"""
        for position in self.positions[:]:
            if position['symbol'] == symbol:
                self._close_position(position, timestamp, price, 'signal')
        
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current portfolio value"""
        # Current capital plus value of open positions
        portfolio_value = self.current_capital
        
        for position in self.positions:
            # Get current position value
            current_price = position.get('current_price', position['entry_price'])
            size = position['size']
            side = position['side']
            entry_price = position['entry_price']
            
            if side == 'long':
                position_value = current_price * size
                unrealized_pnl = (current_price - entry_price) * size
            else:  # short
                position_value = entry_price * size  # Margin for short
                unrealized_pnl = (entry_price - current_price) * size
                
            position['unrealized_pnl'] = unrealized_pnl
            portfolio_value += unrealized_pnl
            
        self.equity_curve.append((timestamp, portfolio_value))
        
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades and not self.equity_curve:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'final_equity': self.initial_capital,
                'total_return': 0.0,
                'annualized_return': 0.0
            }
        
        # Basic trade metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in self.trades if t['pnl'] <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = sum(t['pnl'] for t in self.trades if t['pnl'] <= 0)
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Extract equity values and dates from equity curve
        dates = [ec[0] for ec in self.equity_curve]
        equity_values = [ec[1] for ec in self.equity_curve]
        
        # Calculate drawdown
        max_equity = equity_values[0]
        drawdowns = []
        
        for equity in equity_values:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
            drawdowns.append(drawdown)
            
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calculate returns
        final_equity = equity_values[-1] if equity_values else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calculate trading period in years
        if len(dates) > 1:
            trading_days = (dates[-1] - dates[0]).days
            years = trading_days / 365
            annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        else:
            annualized_return = 0
            
        # Calculate Sharpe ratio
        if len(equity_values) > 1:
            daily_returns = []
            for i in range(1, len(equity_values)):
                daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                daily_returns.append(daily_return)
                
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            # Annualized Sharpe ratio (assuming 252 trading days)
            sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return
        }
        
    def plot_results(self, filename: str = None):
        """Plot backtest results"""
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
            
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Extract dates and equity values
        dates = [ec[0] for ec in self.equity_curve]
        equity_values = [ec[1] for ec in self.equity_curve]
        
        # Plot equity curve
        ax1.plot(dates, equity_values, label='Portfolio Value')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Calculate drawdown
        max_equity = equity_values[0]
        drawdowns = []
        
        for equity in equity_values:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
            drawdowns.append(drawdown)
            
        # Plot drawdown
        ax2.fill_between(dates, drawdowns, color='red', alpha=0.3)
        ax2.plot(dates, drawdowns, color='red', label='Drawdown')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown %')
        ax2.legend()
        ax2.grid(True)
        
        # Plot trade markers
        for trade in self.trades:
            if trade['pnl'] > 0:
                ax1.scatter(trade['exit_time'], trade['exit_price'], 
                          marker='^', color='green', s=50)
            else:
                ax1.scatter(trade['exit_time'], trade['exit_price'], 
                          marker='v', color='red', s=50)
                
        # Plot daily returns
        if len(equity_values) > 1:
            daily_returns = []
            return_dates = []
            
            for i in range(1, len(equity_values)):
                daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                daily_returns.append(daily_return * 100)  # Convert to percentage
                return_dates.append(dates[i])
                
            ax3.bar(return_dates, daily_returns, color=[
                'green' if r > 0 else 'red' for r in daily_returns
            ], alpha=0.7)
            ax3.set_title('Daily Returns')
            ax3.set_ylabel('Return %')
            ax3.grid(True)
            
        # Format x-axis
        fig.autofmt_xdate()
        
        # Add metrics text
        metrics = self._calculate_metrics()
        metrics_text = (
            f"Total Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annual Return: {metrics['annualized_return']:.2%}"
        )
        
        plt.figtext(0.01, 0.01, metrics_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
        plt.close()
        
    def save_results(self, filename: str):
        """Save backtest results to JSON file"""
        if not self.results:
            logger.warning("No results to save")
            return
            
        # Prepare serializable results
        output = {
            'metrics': self.results['metrics'],
            'equity_curve': [(str(dt), val) for dt, val in self.equity_curve],
            'trades': []
        }
        
        # Format trades for serialization
        for trade in self.trades:
            serialized_trade = {
                'symbol': trade['symbol'],
                'entry_time': str(trade['entry_time']),
                'exit_time': str(trade['exit_time']),
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'size': trade['size'],
                'side': trade['side'],
                'pnl': trade['pnl'],
                'pnl_pct': trade['pnl_pct'],
                'reason': trade['reason']
            }
            output['trades'].append(serialized_trade)
            
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=4)
                
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

    # Add this to src/strategy/backtest.py

class WalkForwardOptimizer:
    """
    Walk-forward optimization for trading strategies.
    Helps prevent overfitting by using out-of-sample testing.
    """
    
    def __init__(self, 
               strategy_class,
               data_dict: Dict[str, pd.DataFrame],
               parameter_ranges: Dict[str, Tuple[float, float]],
               base_config: Dict[str, Any] = None):
        """
        Initialize walk-forward optimizer
        
        Args:
            strategy_class: Class of the strategy to optimize
            data_dict: Dictionary of DataFrames with historical data
            parameter_ranges: Dictionary of parameter ranges to test
            base_config: Base configuration for the strategy
        """
        self.strategy_class = strategy_class
        self.data_dict = data_dict
        self.parameter_ranges = parameter_ranges
        self.base_config = base_config or {}
        self.backtest_engine = BacktestEngine()
        self.results = []
        
    def optimize(self, 
               num_folds: int = 5,
               train_size: float = 0.7,
               num_trials: int = 20,
               random_trials: bool = True) -> Dict[str, Any]:
        """
        Run walk-forward optimization
        
        Args:
            num_folds: Number of time periods to test
            train_size: Portion of each fold to use for training
            num_trials: Number of parameter combinations to test in each fold
            random_trials: Whether to use random parameter combinations or grid search
            
        Returns:
            Dict with optimization results and best parameters
        """
        try:
            logger.info("Starting walk-forward optimization")
            
            # Split data into folds
            folds = self._create_time_folds(num_folds)
            
            # Process each fold
            for fold_idx, (train_data, test_data) in enumerate(folds):
                logger.info(f"Processing fold {fold_idx+1}/{num_folds}")
                
                # Generate parameter combinations
                if random_trials:
                    parameter_sets = self._generate_random_parameters(num_trials)
                else:
                    parameter_sets = self._generate_grid_parameters(num_trials)
                
                # Evaluate on training data
                train_results = self._evaluate_parameters(parameter_sets, train_data)
                
                # Get best parameters from training
                best_params, best_metric = self._get_best_parameters(train_results)
                
                # Test on out-of-sample data
                test_results = self._test_parameters(best_params, test_data)
                
                # Store results
                fold_result = {
                    'fold': fold_idx,
                    'best_parameters': best_params,
                    'train_metric': best_metric,
                    'test_metrics': test_results
                }
                
                self.results.append(fold_result)
                
                logger.info(f"Fold {fold_idx+1} - Best train metric: {best_metric:.4f}, "
                           f"Test return: {test_results['total_return']:.2%}")
            
            # Aggregate results
            best_combined_params = self._get_robust_parameters()
            
            logger.info("Walk-forward optimization completed")
            logger.info(f"Best combined parameters: {best_combined_params}")
            
            return {
                'best_parameters': best_combined_params,
                'fold_results': self.results
            }
            
        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {str(e)}")
            raise StrategyError.OptimizationError("Failed to run walk-forward optimization",
                                                details={'error': str(e)})
    
    def _create_time_folds(self, num_folds: int) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]:
        """Create time-based folds for walk-forward optimization"""
        folds = []
        
        # Determine overall date range
        min_dates = []
        max_dates = []
        
        for symbol, df in self.data_dict.items():
            min_dates.append(df.index.min())
            max_dates.append(df.index.max())
            
        global_min_date = min(min_dates)
        global_max_date = max(max_dates)
        
        # Calculate time range
        total_days = (global_max_date - global_min_date).days
        fold_days = total_days // num_folds
        
        # Create folds
        for i in range(num_folds):
            # Calculate fold dates
            fold_start = global_min_date + timedelta(days=i * fold_days)
            fold_end = fold_start + timedelta(days=fold_days)
            
            # Calculate train/test split
            train_end = fold_start + timedelta(days=int(fold_days * 0.7))
            
            # Create train and test dataframes
            train_data = {}
            test_data = {}
            
            for symbol, df in self.data_dict.items():
                # Filter for fold dates
                fold_df = df[(df.index >= fold_start) & (df.index <= fold_end)].copy()
                
                if not fold_df.empty:
                    # Split into train and test
                    train_df = fold_df[fold_df.index <= train_end].copy()
                    test_df = fold_df[fold_df.index > train_end].copy()
                    
                    if not train_df.empty:
                        train_data[symbol] = train_df
                    if not test_df.empty:
                        test_data[symbol] = test_df
            
            if train_data and test_data:
                folds.append((train_data, test_data))
        
        return folds
    
    def _generate_random_parameters(self, num_trials: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations within ranges"""
        param_sets = []
        
        for _ in range(num_trials):
            params = self.base_config.copy()
            
            for param_name, (min_val, max_val) in self.parameter_ranges.items():
                # Handle different parameter types
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                elif isinstance(min_val, bool) or isinstance(max_val, bool):
                    # Boolean parameter
                    params[param_name] = bool(np.random.randint(0, 2))
                else:
                    # Float parameter
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            param_sets.append(params)
        
        return param_sets
    
    def _generate_grid_parameters(self, max_trials: int) -> List[Dict[str, Any]]:
        """Generate grid-based parameter combinations"""
        # Determine number of points for each parameter
        num_params = len(self.parameter_ranges)
        points_per_param = max(2, int(max_trials ** (1/num_params)))
        
        # Generate values for each parameter
        param_values = {}
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                param_values[param_name] = np.linspace(min_val, max_val, 
                                                     min(max_val - min_val + 1, points_per_param), 
                                                     dtype=int).tolist()
            elif isinstance(min_val, bool) or isinstance(max_val, bool):
                # Boolean parameter
                param_values[param_name] = [False, True]
            else:
                # Float parameter
                param_values[param_name] = np.linspace(min_val, max_val, 
                                                    points_per_param).tolist()
        
        # Generate combinations
        param_names = list(param_values.keys())
        combinations = []
        
        def generate_combinations(param_idx, current_params):
            if param_idx == len(param_names) or len(combinations) >= max_trials:
                return
            
            param_name = param_names[param_idx]
            for value in param_values[param_name]:
                current_params[param_name] = value
                
                if param_idx == len(param_names) - 1:
                    combinations.append(current_params.copy())
                    if len(combinations) >= max_trials:
                        return
                else:
                    generate_combinations(param_idx + 1, current_params)
        
        generate_combinations(0, self.base_config.copy())
        
        return combinations[:max_trials]
    
    def _evaluate_parameters(self, parameter_sets: List[Dict[str, Any]],
                          data_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Evaluate multiple parameter sets on training data"""
        results = []
        
        for params in parameter_sets:
            try:
                # Create strategy with parameters
                strategy = self.strategy_class(params)
                
                # Run backtest
                backtest_result = self.backtest_engine.run_backtest(
                    strategy,
                    data_dict,
                    transaction_costs=True
                )
                
                # Calculate combined metric
                metrics = backtest_result['metrics']
                
                # Create a combined metric that balances return and risk
                combined_metric = (
                    metrics['total_return'] * 0.4 +
                    metrics['sharpe_ratio'] * 0.3 +
                    metrics['win_rate'] * 0.1 -
                    metrics['max_drawdown'] * 0.2
                )
                
                # Store results
                result = {
                    'parameters': params,
                    'metrics': metrics,
                    'combined_metric': combined_metric
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating parameters: {str(e)}")
                continue
        
        return results
    
    def _get_best_parameters(self, results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """Get best parameters based on combined metric"""
        if not results:
            return self.base_config, 0.0
            
        # Sort by combined metric (higher is better)
        sorted_results = sorted(results, key=lambda x: x['combined_metric'], reverse=True)
        
        return sorted_results[0]['parameters'], sorted_results[0]['combined_metric']
    
    def _test_parameters(self, parameters: Dict[str, Any],
                       data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Test parameters on out-of-sample data"""
        try:
            # Create strategy with parameters
            strategy = self.strategy_class(parameters)
            
            # Run backtest
            backtest_result = self.backtest_engine.run_backtest(
                strategy,
                data_dict,
                transaction_costs=True
            )
            
            return backtest_result['metrics']
            
        except Exception as e:
            logger.error(f"Error testing parameters: {str(e)}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 1.0
            }
    
    def _get_robust_parameters(self) -> Dict[str, Any]:
        """Get robust parameters by averaging across folds"""
        if not self.results:
            return self.base_config
            
        # Initialize parameter accumulator
        combined_params = {}
        
        # Calculate weighted average based on test performance
        total_weight = 0
        
        for fold_result in self.results:
            # Use test return as weight
            weight = max(0.1, fold_result['test_metrics']['total_return'] + 0.5)
            total_weight += weight
            
            # Add weighted parameters
            for param_name, param_value in fold_result['best_parameters'].items():
                if param_name not in combined_params:
                    combined_params[param_name] = 0
                    
                combined_params[param_name] += param_value * weight
        
        # Normalize by total weight
        for param_name in combined_params:
            # Check if this is an integer parameter
            is_int = any(isinstance(fold['best_parameters'].get(param_name), int) 
                       for fold in self.results)
            
            # Check if this is a boolean parameter
            is_bool = any(isinstance(fold['best_parameters'].get(param_name), bool)
                        for fold in self.results)
            
            # Normalize
            if total_weight > 0:
                combined_params[param_name] /= total_weight
                
                # Convert back to original type
                if is_int:
                    combined_params[param_name] = int(round(combined_params[param_name]))
                elif is_bool:
                    combined_params[param_name] = combined_params[param_name] >= 0.5
        
        return combined_params    