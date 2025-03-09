#!/usr/bin/env python
"""
Results Interpreter for Trading Strategy Backtests

This script loads and analyzes the JSON results from multiple backtest runs,
providing comparisons and insights to help determine the most effective strategy.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Any

def load_results(results_dir: str = 'results') -> Dict[str, Dict]:
    """Load all JSON result files from the results directory"""
    results = {}
    
    results_path = Path(results_dir)
    if not results_path.exists() or not results_path.is_dir():
        print(f"Error: Results directory '{results_dir}' not found")
        return {}
    
    for file_path in results_path.glob('*.json'):
        try:
            with open(file_path, 'r') as f:
                result_data = json.load(f)
                
                # Extract strategy name from filename
                filename = file_path.stem
                strategy_name = filename.split('_')[0]
                
                results[filename] = {
                    'strategy': strategy_name,
                    'timestamp': file_path.stat().st_mtime,
                    'data': result_data
                }
                
                print(f"Loaded results from {file_path}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def parse_equity_curve(results: Dict) -> pd.DataFrame:
    """Convert equity curve data to DataFrame"""
    equity_data = []
    
    for filename, result in results.items():
        if 'equity_curve' in result['data']:
            equity_points = result['data']['equity_curve']
            
            # Create DataFrame
            points = []
            for date_str, value in equity_points:
                try:
                    # Handle different date formats
                    try:
                        date = datetime.fromisoformat(date_str)
                    except ValueError:
                        date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    
                    points.append({
                        'strategy': result['strategy'],
                        'date': date, 
                        'equity': value
                    })
                except Exception as e:
                    print(f"Error parsing date {date_str}: {e}")
            
            if points:
                equity_data.extend(points)
    
    if not equity_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(equity_data)
    df = df.sort_values('date')
    return df

def analyze_trades(results: Dict) -> pd.DataFrame:
    """Analyze trade data across strategies"""
    all_trades = []
    
    for filename, result in results.items():
        if 'trades' in result['data']:
            trades = result['data']['trades']
            
            for trade in trades:
                trade_data = {
                    'strategy': result['strategy'],
                    'symbol': trade.get('symbol', ''),
                    'side': trade.get('side', ''),
                    'entry_price': trade.get('entry_price', 0),
                    'exit_price': trade.get('exit_price', 0),
                    'size': trade.get('size', 0),
                    'pnl': trade.get('pnl', 0),
                    'pnl_pct': trade.get('pnl_pct', 0),
                    'entry_time': trade.get('entry_time', ''),
                    'exit_time': trade.get('exit_time', ''),
                    'duration': 0,  # Will calculate below
                    'reason': trade.get('reason', '')
                }
                
                # Calculate trade duration if possible
                try:
                    entry_time = datetime.fromisoformat(trade_data['entry_time'])
                    exit_time = datetime.fromisoformat(trade_data['exit_time'])
                    duration = exit_time - entry_time
                    trade_data['duration'] = duration.total_seconds() / 3600  # Duration in hours
                except (ValueError, TypeError):
                    pass
                
                all_trades.append(trade_data)
    
    if not all_trades:
        return pd.DataFrame()
    
    return pd.DataFrame(all_trades)

def compare_metrics(results: Dict) -> pd.DataFrame:
    """Compare key metrics across strategies"""
    metrics = []
    
    for filename, result in results.items():
        if 'metrics' in result['data']:
            result_metrics = result['data']['metrics']
            
            # Convert metrics to percentage where applicable
            metrics_row = {
                'strategy': result['strategy'],
                'filename': filename,
                'total_trades': result_metrics.get('total_trades', 0),
                'win_rate': result_metrics.get('win_rate', 0) * 100,  # Convert to percentage
                'profit_factor': result_metrics.get('profit_factor', 0),
                'total_return': result_metrics.get('total_return', 0) * 100,  # Convert to percentage
                'annualized_return': result_metrics.get('annualized_return', 0) * 100,  # Convert to percentage
                'sharpe_ratio': result_metrics.get('sharpe_ratio', 0),
                'max_drawdown': result_metrics.get('max_drawdown', 0) * 100,  # Convert to percentage
                'final_equity': result_metrics.get('final_equity', 0)
            }
            
            metrics.append(metrics_row)
    
    if not metrics:
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics)
    
    # Sort by annualized return (or total return if annualized not available)
    if 'annualized_return' in df.columns and not df['annualized_return'].isna().all():
        df = df.sort_values('annualized_return', ascending=False)
    else:
        df = df.sort_values('total_return', ascending=False)
    
    return df

def analyze_market_conditions(equity_df: pd.DataFrame, trade_df: pd.DataFrame) -> Dict:
    """Analyze how strategies perform in different market conditions"""
    if equity_df.empty or trade_df.empty:
        return {}
    
    # Identify market conditions based on price movement
    # This is a simplistic approach - could be enhanced with actual market data
    results = {}
    
    try:
        # Group trades by strategy
        strategies = trade_df['strategy'].unique()
        
        for strategy in strategies:
            strategy_trades = trade_df[trade_df['strategy'] == strategy]
            
            # Calculate winning and losing trade statistics
            winning_trades = strategy_trades[strategy_trades['pnl'] > 0]
            losing_trades = strategy_trades[strategy_trades['pnl'] <= 0]
            
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
            
            # Duration analysis
            avg_win_duration = winning_trades['duration'].mean() if not winning_trades.empty else 0
            avg_loss_duration = losing_trades['duration'].mean() if not losing_trades.empty else 0
            
            # Trade distribution by size
            trade_size_bins = pd.qcut(strategy_trades['pnl'].abs(), 4, duplicates='drop')
            size_performance = strategy_trades.groupby(trade_size_bins)['pnl'].agg(['mean', 'count'])
            
            results[strategy] = {
                'win_count': len(winning_trades),
                'loss_count': len(losing_trades),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_win_duration': avg_win_duration,
                'avg_loss_duration': avg_loss_duration,
                'trade_size_performance': size_performance.to_dict() if not size_performance.empty else {},
                'consecutive_wins': find_consecutive_count(strategy_trades['pnl'] > 0),
                'consecutive_losses': find_consecutive_count(strategy_trades['pnl'] <= 0)
            }
    
    except Exception as e:
        print(f"Error analyzing market conditions: {e}")
    
    return results

def find_consecutive_count(series):
    """Find the longest streak of True values in a boolean series"""
    if series.empty:
        return 0
    
    # Convert to 0/1
    values = series.astype(int).values
    
    # Identify the starts of runs
    runs = np.diff(np.hstack(([0], values, [0])))
    
    # Positive run starts are 1, ends are -1
    run_starts = np.where(runs == 1)[0]
    run_ends = np.where(runs == -1)[0]
    
    # Calculate lengths of runs
    run_lengths = run_ends - run_starts
    
    # Return the maximum run length
    return max(run_lengths) if len(run_lengths) > 0 else 0

def plot_equity_curves(equity_df: pd.DataFrame, save_path: str = None):
    """Plot equity curves for all strategies"""
    if equity_df.empty:
        print("No equity data to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves for each strategy
    for strategy, group in equity_df.groupby('strategy'):
        plt.plot(group['date'], group['equity'], label=strategy)
    
    plt.title('Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved equity curve plot to {save_path}")
    else:
        plt.show()

def plot_drawdowns(equity_df: pd.DataFrame, save_path: str = None):
    """Plot drawdowns for all strategies"""
    if equity_df.empty:
        print("No equity data to plot drawdowns")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Calculate and plot drawdowns for each strategy
    for strategy, group in equity_df.groupby('strategy'):
        # Calculate drawdown
        equity = group['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100  # Convert to percentage
        
        plt.plot(group['date'], drawdown, label=strategy)
    
    plt.title('Strategy Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved drawdown plot to {save_path}")
    else:
        plt.show()

def plot_trade_distribution(trade_df: pd.DataFrame, save_path: str = None):
    """Plot distribution of trade PnL by strategy"""
    if trade_df.empty:
        print("No trade data to plot distribution")
        return
    
    # Create subplots for each strategy
    strategies = trade_df['strategy'].unique()
    fig, axes = plt.subplots(len(strategies), 1, figsize=(12, 4 * len(strategies)))
    
    # If only one strategy, convert axes to a list
    if len(strategies) == 1:
        axes = [axes]
    
    for i, strategy in enumerate(strategies):
        strategy_trades = trade_df[trade_df['strategy'] == strategy]
        
        # Plot histogram of PnL
        axes[i].hist(strategy_trades['pnl'], bins=50)
        axes[i].set_title(f'{strategy} - Trade PnL Distribution')
        axes[i].set_xlabel('PnL')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved trade distribution plot to {save_path}")
    else:
        plt.show()

def print_summary(metrics_df: pd.DataFrame, market_analysis: Dict):
    """Print summary of strategy performance"""
    if metrics_df.empty:
        print("No metrics data to summarize")
        return
    
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    
    # Print overall metrics
    print("\nPerformance Metrics:")
    print(metrics_df.to_string(index=False))
    
    print("\nDetailed Strategy Analysis:")
    for strategy, analysis in market_analysis.items():
        print(f"\n{strategy}:")
        print(f"  Winning Trades: {analysis['win_count']}")
        print(f"  Losing Trades: {analysis['loss_count']}")
        print(f"  Average Win: ${analysis['avg_win']:.2f}")
        print(f"  Average Loss: ${analysis['avg_loss']:.2f}")
        print(f"  Average Win Duration: {analysis['avg_win_duration']:.2f} hours")
        print(f"  Average Loss Duration: {analysis['avg_loss_duration']:.2f} hours")
        print(f"  Max Consecutive Wins: {analysis['consecutive_wins']}")
        print(f"  Max Consecutive Losses: {analysis['consecutive_losses']}")
    
    # Find best performing strategy
    if not metrics_df.empty:
        best_strategy = metrics_df.iloc[0]['strategy']
        best_return = metrics_df.iloc[0]['total_return']
        print(f"\nBest performing strategy: {best_strategy} with {best_return:.2f}% total return")
    
    print("\nRecommendations:")
    if not metrics_df.empty:
        best_row = metrics_df.iloc[0]
        strategy = best_row['strategy']
        
        if best_row['total_return'] > 0 and best_row['sharpe_ratio'] > 1:
            print(f"- {strategy} shows promising results and could be suitable for paper trading")
        else:
            print(f"- All strategies need further optimization before paper trading")
        
        if best_row['max_drawdown'] > 20:
            print(f"- {strategy} has a high maximum drawdown ({best_row['max_drawdown']:.2f}%) - consider tighter risk controls")
        
        if best_row['win_rate'] < 50:
            print(f"- {strategy} has a low win rate ({best_row['win_rate']:.2f}%) - need better entry/exit rules")
    
    print("="*80)

def generate_report(results_dir: str = 'results', output_dir: str = 'analysis'):
    """Generate comprehensive analysis report from backtest results"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load results
    print(f"Loading backtest results from {results_dir}")
    results = load_results(results_dir)
    
    if not results:
        print("No results found to analyze")
        return
    
    # Parse data
    print("Parsing equity curves")
    equity_df = parse_equity_curve(results)
    
    print("Analyzing trades")
    trade_df = analyze_trades(results)
    
    print("Comparing metrics")
    metrics_df = compare_metrics(results)
    
    print("Analyzing market conditions")
    market_analysis = analyze_market_conditions(equity_df, trade_df)
    
    # Generate visualizations
    if not equity_df.empty:
        print("Generating equity curve plot")
        plot_equity_curves(equity_df, save_path=f"{output_dir}/equity_curves.png")
        
        print("Generating drawdown plot")
        plot_drawdowns(equity_df, save_path=f"{output_dir}/drawdowns.png")
    
    if not trade_df.empty:
        print("Generating trade distribution plot")
        plot_trade_distribution(trade_df, save_path=f"{output_dir}/trade_distribution.png")
    
    # Print summary
    print_summary(metrics_df, market_analysis)
    
    # Export data to CSV
    if not metrics_df.empty:
        metrics_df.to_csv(f"{output_dir}/metrics_comparison.csv", index=False)
        print(f"Exported metrics to {output_dir}/metrics_comparison.csv")
    
    if not trade_df.empty:
        trade_df.to_csv(f"{output_dir}/trade_analysis.csv", index=False)
        print(f"Exported trade analysis to {output_dir}/trade_analysis.csv")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze backtest results and compare strategies")
    parser.add_argument("--results-dir", default="results", help="Directory containing backtest result files")
    parser.add_argument("--output-dir", default="analysis", help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_dir)