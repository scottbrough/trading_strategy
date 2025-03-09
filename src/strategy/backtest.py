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
        self.positions = []  # Make sure this is an empty list, not None
        self.trades = []
        self.equity_curve = [(datetime.now(), self.initial_capital)]
        logger.info(f"Initialized backtest with capital: {self.initial_capital}")
            
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
            # Calculate technical indicators
            from ..data.processor import DataProcessor
            data_processor = DataProcessor()
            df_copy = data_processor.process_ohlcv(df_copy, 'unknown')  # 'unknown' is a placeholder   
            processed_data[symbol] = df_copy
            
        if not processed_data:
            raise ValueError("No valid data after preparation")
            
        return processed_data
        
    def _generate_signals(self, strategy: Any,
                        data_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate trading signals from strategy"""
        all_signals = []
        
        # Initialize or update strategy parameters once
        if not hasattr(strategy, 'params'):
            strategy.params = {}
        strategy.params['current_capital'] = self.current_capital
        
        # Generate signals for each symbol
        for symbol, df in data_dict.items():
            try:
                # Generate signals for this symbol
                signals = strategy.generate_signals(df)
                
                # Add symbol to each signal if not present
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
            last_dates = [data_dict[pos['symbol']].index[-1] for pos in self.positions 
                       if pos['symbol'] in data_dict]
            if last_dates:
                last_date = max(last_dates)
                self._update_equity_curve(last_date)
        
    def _open_position(self, symbol: str, timestamp: datetime, 
                     price: float, size: float, side: str):
        """Open a new position"""
        # Check if we have enough capital
        position_value = price * size
        max_position_size = self.config.get('max_position_size', 0.2)
        
        if position_value > self.current_capital * max_position_size:
            size = (self.current_capital * max_position_size) / price
            
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
        # Start with current cash
        portfolio_value = self.current_capital
        total_unrealized_pnl = 0
        
        # Only calculate position values if we have positions
        if self.positions:
            logger.debug(f"Calculating equity with {len(self.positions)} positions")
            for position in self.positions:
                # Get current position value
                current_price = position.get('current_price', position['entry_price'])
                size = position['size']
                side = position['side']
                entry_price = position['entry_price']
                
                # Calculate unrealized P&L
                if side == 'long':
                    unrealized_pnl = (current_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - current_price) * size
                    
                position['unrealized_pnl'] = unrealized_pnl
                total_unrealized_pnl += unrealized_pnl
        
        # Calculate total portfolio value (cash + unrealized P&L)
        portfolio_value = self.current_capital + total_unrealized_pnl
        
        logger.debug(f"Equity update: capital={self.current_capital}, unrealized_pnl={total_unrealized_pnl}, total={portfolio_value}")
        
        # Sanity check - portfolio value should equal current_capital if no positions
        if not self.positions:
            portfolio_value = self.current_capital
        # Safety check - if no positions and no trades, portfolio should equal initial capital
        if not self.positions and not self.trades and portfolio_value != self.initial_capital:
            logger.warning(f"Portfolio value {portfolio_value} differs from initial capital {self.initial_capital} with no positions/trades - fixing")
            portfolio_value = self.initial_capital  # Fix the value
        
        # Add to equity curve
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


