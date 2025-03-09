"""
Parameter optimization module using genetic algorithms and other advanced methods.
Implements walk-forward optimization and cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from deap import base, creator, tools, algorithms
import multiprocessing
from datetime import datetime
import json
from sklearn.model_selection import TimeSeriesSplit
import optuna
from functools import partial

from ..core.logger import log_manager
from ..core.config import config
from ..core.exceptions import StrategyError
from .backtest import BacktestEngine

logger = log_manager.get_logger(__name__)

class StrategyOptimizer:
    def __init__(self, strategy_class: type, data: Dict[str, pd.DataFrame]):
        """Initialize optimizer with strategy class and data"""
        self.strategy_class = strategy_class
        self.data = data
        self.backtest_engine = BacktestEngine()
        self.config = config.get_optimization_params()
        self.setup_genetic_algorithm()
    
    def setup_genetic_algorithm(self):
        """Initialize genetic algorithm components"""
        try:
            # Clear any existing DEAP classes
            if 'FitnessMax' in creator.__dict__:
                del creator.FitnessMax
            if 'Individual' in creator.__dict__:
                del creator.Individual
            
            # Create new fitness and individual classes
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Set up toolbox
            self.toolbox = base.Toolbox()
            
            # Register genetic operators
            self.toolbox.register("attr_float", np.random.uniform, 0, 1)
            self.toolbox.register("individual", self._create_individual)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("evaluate", self._evaluate_individual)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", self._custom_mutation)
            self.toolbox.register("select", tools.selTournament, tournsize=3)
            
        except Exception as e:
            logger.error(f"Error setting up genetic algorithm: {str(e)}")
            raise StrategyError.OptimizationError("Failed to setup genetic algorithm")
    
    def _create_individual(self):
        """Create individual with parameter constraints"""
        params = {
            'rsi_period': np.random.randint(10, 30),
            'macd_fast': np.random.randint(8, 20),
            'macd_slow': np.random.randint(20, 40),
            'macd_signal': np.random.randint(7, 15),
            'bb_period': np.random.randint(15, 30),
            'bb_stddev': np.random.uniform(1.5, 3.0),
            'stop_loss': np.random.uniform(0.01, 0.05),
            'take_profit': np.random.uniform(0.02, 0.1),
            'risk_factor': np.random.uniform(0.01, 0.05),
            'momentum_factor': np.random.uniform(0.5, 2.0),
            'trend_strength': np.random.uniform(0.3, 0.7)
        }
        return creator.Individual([v for v in params.values()])
    
    def _custom_mutation(self, individual: List, indpb: float = 0.2) -> Tuple[List]:
        """Custom mutation operator with parameter constraints"""
        try:
            for i in range(len(individual)):
                if np.random.random() < indpb:
                    if i in [0, 1, 2, 3, 4]:  # Integer parameters
                        individual[i] = int(individual[i] * (1 + np.random.normal(0, 0.2)))
                    else:  # Float parameters
                        individual[i] *= (1 + np.random.normal(0, 0.2))
                    
                    # Ensure parameters stay within bounds
                    individual[i] = max(self.config['param_bounds'][i][0],
                                     min(self.config['param_bounds'][i][1], individual[i]))
            
            return individual,
            
        except Exception as e:
            logger.error(f"Mutation error: {str(e)}")
            return individual,
    
    def _evaluate_individual(self, individual: List) -> Tuple[float]:
        """Evaluate individual using walk-forward analysis"""
        try:
            params = self._decode_individual(individual)
            
            # Initialize metrics storage
            results = []
            
            # Walk-forward analysis
            tscv = TimeSeriesSplit(n_splits=5)
            
            for symbol, data in self.data.items():
                for train_idx, test_idx in tscv.split(data):
                    train_data = data.iloc[train_idx]
                    test_data = data.iloc[test_idx]
                    
                    # Run backtest
                    strategy = self.strategy_class(params)
                    metrics = self.backtest_engine.run_backtest(strategy, test_data)
                    
                    # Calculate fitness score
                    fitness = self._calculate_fitness(metrics)
                    results.append(fitness)
            
            # Return average fitness across all periods and symbols
            return (np.mean(results),)
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return (-np.inf,)
    
    def _decode_individual(self, individual: List) -> Dict:
        """Convert individual to parameter dictionary"""
        param_names = [
            'rsi_period', 'macd_fast', 'macd_slow', 'macd_signal',
            'bb_period', 'bb_stddev', 'stop_loss', 'take_profit',
            'risk_factor', 'momentum_factor', 'trend_strength'
        ]
        return dict(zip(param_names, individual))
    
    def _calculate_fitness(self, metrics: Dict) -> float:
        """Calculate comprehensive fitness score from backtest metrics"""
        
        # Extract key metrics (with safety defaults)
        total_return = metrics.get('total_return', 0) 
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        sortino_ratio = metrics.get('sortino_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        num_trades = metrics.get('total_trades', 0)
        
        # Penalize excessive drawdown with exponential scaling
        drawdown_penalty = np.exp(-5 * max_drawdown)
        
        # Penalize too frequent trading
        trade_frequency_penalty = min(1.0, 30 / max(num_trades, 1)) if num_trades > 30 else 1.0
        
        # Penalize low win rate
        low_win_rate_penalty = 1.0 if win_rate > 0.4 else win_rate / 0.4
        
        # Combine metrics with balanced weights
        fitness = (
            0.3 * total_return +
            0.2 * sharpe_ratio +
            0.15 * sortino_ratio +
            0.15 * drawdown_penalty +
            0.1 * profit_factor +
            0.05 * win_rate +
            0.05 * trade_frequency_penalty
        ) * low_win_rate_penalty
        
        return max(fitness, 0.001)  # Ensure non-zero fitness
    
    def optimize_genetic(self, 
                       population_size: int = 50,
                       generations: int = 30,
                       checkpoint_freq: int = 5) -> Dict:
        """Run genetic algorithm optimization"""
        try:
            logger.info("Starting genetic algorithm optimization")
            
            # Initialize population
            population = self.toolbox.population(n=population_size)
            
            # Statistics setup
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Hall of Fame
            hof = tools.HallOfFame(5)
            
            # Run evolution with checkpoints
            for gen in range(generations):
                # Select the next generation individuals
                offspring = algorithms.varAnd(
                    population, self.toolbox,
                    cxpb=0.7,  # crossover probability
                    mutpb=0.3   # mutation probability
                )
                
                # Evaluate the individuals
                fits = self.toolbox.map(self.toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                
                # Update population and hall of fame
                population = self.toolbox.select(offspring, len(population))
                hof.update(population)
                
                # Save checkpoint
                if gen % checkpoint_freq == 0:
                    self._save_checkpoint(gen, population, hof)
                
                # Log progress
                record = stats.compile(population)
                logger.info(f"Generation {gen}: {record}")
            
            # Get best parameters
            best_individual = hof[0]
            best_params = self._decode_individual(best_individual)
            
            return best_params
            
        except Exception as e:
            logger.error(f"Genetic optimization error: {str(e)}")
            raise StrategyError.OptimizationError("Failed to optimize parameters")
    
    def optimize_bayesian(self, n_trials: int = 100) -> Dict:
        """Run Bayesian optimization using Optuna"""
        try:
            def objective(trial):
                params = {
                    'rsi_period': trial.suggest_int('rsi_period', 10, 30),
                    'macd_fast': trial.suggest_int('macd_fast', 8, 20),
                    'macd_slow': trial.suggest_int('macd_slow', 20, 40),
                    'macd_signal': trial.suggest_int('macd_signal', 7, 15),
                    'bb_period': trial.suggest_int('bb_period', 15, 30),
                    'bb_stddev': trial.suggest_float('bb_stddev', 1.5, 3.0),
                    'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
                    'take_profit': trial.suggest_float('take_profit', 0.02, 0.1),
                    'risk_factor': trial.suggest_float('risk_factor', 0.01, 0.05),
                    'momentum_factor': trial.suggest_float('momentum_factor', 0.5, 2.0),
                    'trend_strength': trial.suggest_float('trend_strength', 0.3, 0.7)
                }
                
                strategy = self.strategy_class(params)
                metrics = self.backtest_engine.run_backtest(strategy, self.data)
                return self._calculate_fitness(metrics)
            
            # Create and run study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Bayesian optimization error: {str(e)}")
            raise StrategyError.OptimizationError("Failed to optimize parameters")
    
    def _save_checkpoint(self, generation: int, population: List, hof: tools.HallOfFame):
        """Save optimization checkpoint"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                'generation': generation,
                'population': [ind[:] for ind in population],
                'hof': [ind[:] for ind in hof],
                'params': [self._decode_individual(ind) for ind in hof]
            }
            
            with open(f'checkpoints/optimization_{timestamp}.json', 'w') as f:
                json.dump(checkpoint, f, indent=4)
                
        except Exception as e:
            logger.error(f"Checkpoint save error: {str(e)}")
    
    def load_checkpoint(self, filepath: str) -> Tuple[int, List, tools.HallOfFame]:
        """Load optimization checkpoint"""
        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
            
            population = [creator.Individual(ind) for ind in checkpoint['population']]
            hof = tools.HallOfFame(5)
            hof.update([creator.Individual(ind) for ind in checkpoint['hof']])
            
            return checkpoint['generation'], population, hof
            
        except Exception as e:
            logger.error(f"Checkpoint load error: {str(e)}")

    def optimize_strategy(strategy_class, data, initial_params):
        """Two-stage optimization process"""
        # Stage 1: Broad search with grid or random search
        coarse_optimizer = StrategyOptimizer(strategy_class, data)
        
        # Define wide parameter ranges for initial search
        broad_parameter_ranges = {
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'adx_threshold': (15, 35),
            # Add more parameters with wide ranges
        }
        
        # Do coarse optimization first
        coarse_results = coarse_optimizer.optimize_bayesian(
            parameter_ranges=broad_parameter_ranges,
            n_trials=50
        )
        
        # Stage 2: Fine-tuning with narrower ranges
        narrow_ranges = {}
        for param, value in coarse_results['best_parameters'].items():
            # Create narrow range around best value
            # For integers:
            if isinstance(value, int):
                narrow_ranges[param] = (max(value - 5, broad_parameter_ranges[param][0]), 
                                    min(value + 5, broad_parameter_ranges[param][1]))
            # For floats:
            else:
                narrow_ranges[param] = (max(value * 0.8, broad_parameter_ranges[param][0]), 
                                    min(value * 1.2, broad_parameter_ranges[param][1]))
        
        # Fine optimization with genetic algorithm
        fine_optimizer = StrategyOptimizer(strategy_class, data)
        final_results = fine_optimizer.optimize_genetic(
            parameter_ranges=narrow_ranges,
            population_size=50,
            generations=30
        )
        
        return final_results
    
    def _create_time_folds(self, num_folds: int) -> List[Tuple]:
        """Create time-based folds that capture different market regimes"""
        
        # Define market regimes based on volatility or trend characteristics
        def detect_regime(df):
            # Calculate a regime indicator (e.g., volatility)
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            
            # Classify periods
            if volatility.mean() > 0.5:  # High volatility threshold
                return 'volatile'
            elif (df['close'].iloc[-1] / df['close'].iloc[0] - 1) > 0.2:  # Strong trend
                return 'trending'
            else:
                return 'ranging'
        
        # Analyze each time segment for regime classification
        segments = []
        
        # First, divide the data into more segments than folds
        window_size = len(self.data_dict[list(self.data_dict.keys())[0]]) // (num_folds * 2)
        
        for symbol, df in self.data_dict.items():
            for i in range(0, len(df) - window_size, window_size):
                segment_df = df.iloc[i:i+window_size]
                if not segment_df.empty:
                    segments.append({
                        'start_idx': i,
                        'end_idx': i + window_size,
                        'regime': detect_regime(segment_df)
                    })
        
        # Ensure we have segments from each regime type
        regimes = set(s['regime'] for s in segments)
        
        # Create folds that contain a mix of regimes
        folds = []
        for _ in range(num_folds):
            # Select segments to ensure representation of each regime
            train_segments = []
            test_segments = []
            
            for regime in regimes:
                regime_segments = [s for s in segments if s['regime'] == regime]
                if regime_segments:
                    # Use 70% for training, 30% for testing
                    split_idx = int(len(regime_segments) * 0.7)
                    train_segments.extend(regime_segments[:split_idx])
                    test_segments.extend(regime_segments[split_idx:])
            
            # Create train and test data
            train_data = {}
            test_data = {}
            
            for symbol, df in self.data_dict.items():
                train_indices = []
                test_indices = []
                
                for segment in train_segments:
                    train_indices.extend(range(segment['start_idx'], segment['end_idx']))
                
                for segment in test_segments:
                    test_indices.extend(range(segment['start_idx'], segment['end_idx']))
                
                train_data[symbol] = df.iloc[train_indices]
                test_data[symbol] = df.iloc[test_indices]
            
            folds.append((train_data, test_data))
        
        return folds
    
class StrategyEvaluator:
    """Comprehensive strategy evaluation framework"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], initial_capital: float = 10000):
        self.data_dict = data_dict
        self.initial_capital = initial_capital
        self.backtest_engine = BacktestEngine()
        
    def evaluate_strategy(self, strategy, name: str = None) -> Dict:
        """Evaluate a strategy with comprehensive metrics"""
        results = self.backtest_engine.run_backtest(
            strategy,
            self.data_dict,
            initial_capital=self.initial_capital
        )
        
        # Extract standard metrics
        metrics = results['metrics']
        
        # Add additional metrics
        enhanced_metrics = self._calculate_enhanced_metrics(results)
        metrics.update(enhanced_metrics)
        
        # Perform market regime analysis
        regime_performance = self._analyze_regime_performance(results, strategy)
        
        # Monte Carlo analysis
        monte_carlo_results = self._perform_monte_carlo(results['trades'])
        
        # Return comprehensive evaluation
        return {
            'strategy_name': name or strategy.__class__.__name__,
            'metrics': metrics,
            'regime_performance': regime_performance,
            'monte_carlo': monte_carlo_results,
            'trade_stats': self._analyze_trade_stats(results['trades']),
            'robustness_score': self._calculate_robustness_score(metrics, monte_carlo_results)
        }
    
    def _calculate_enhanced_metrics(self, results: Dict) -> Dict:
        """Calculate additional performance metrics"""
        metrics = {}
        trades = results['trades']
        equity_curve = np.array([point[1] for point in results['equity_curve']])
        
        # Calculate additional metrics
        if len(equity_curve) > 1:
            # Recovery factor (return / max drawdown)
            if results['metrics']['max_drawdown'] > 0:
                metrics['recovery_factor'] = results['metrics']['total_return'] / results['metrics']['max_drawdown']
            
            # Ulcer index (measure of drawdown severity)
            underwater = np.maximum.accumulate(equity_curve) - equity_curve
            underwater_pct = underwater / np.maximum.accumulate(equity_curve)
            metrics['ulcer_index'] = np.sqrt(np.mean(np.square(underwater_pct)))
            
            # Calmar ratio (annual return / max drawdown)
            metrics['calmar_ratio'] = results['metrics']['annualized_return'] / max(results['metrics']['max_drawdown'], 0.01)
            
            # Stability of returns (R-squared of equity curve against linear regression)
            x = np.arange(len(equity_curve))
            slope, intercept = np.polyfit(x, equity_curve, 1)
            pred_y = slope * x + intercept
            metrics['r_squared'] = 1 - np.sum((equity_curve - pred_y) ** 2) / np.sum((equity_curve - np.mean(equity_curve)) ** 2)
        
        return metrics
    
    def _analyze_regime_performance(self, results: Dict, strategy) -> Dict:
        """Analyze performance across different market regimes"""
        regimes = {}
        
        # Define market regimes
        for symbol, df in self.data_dict.items():
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            
            df['regime'] = 'normal'
            df.loc[volatility > volatility.quantile(0.7), 'regime'] = 'volatile'
            df.loc[volatility < volatility.quantile(0.3), 'regime'] = 'low_vol'
            
            # Identify trending periods
            if 'adx' in df.columns:
                df.loc[df['adx'] > 30, 'regime'] = 'trending'
        
        # Match trades to regimes
        for trade in results['trades']:
            entry_time = trade['entry_time']
            for symbol, df in self.data_dict.items():
                # Find regime at entry time
                regime_at_entry = df[df.index <= entry_time]['regime'].iloc[-1] if len(df[df.index <= entry_time]) > 0 else 'unknown'
                
                if regime_at_entry not in regimes:
                    regimes[regime_at_entry] = []
                
                regimes[regime_at_entry].append(trade)
        
        # Calculate performance metrics for each regime
        regime_performance = {}
        for regime, trades in regimes.items():
            if not trades:
                continue
                
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
            avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
            
            regime_performance[regime] = {
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl
            }
        
        return regime_performance
    
    def _perform_monte_carlo(self, trades: List[Dict], iterations: int = 1000) -> Dict:
        """Perform Monte Carlo simulation to assess strategy robustness"""
        if not trades:
            return {'worst_drawdown': 0, 'var_95': 0, 'expected_return': 0}
        
        # Extract trade returns
        returns = [t['pnl'] / (t['entry_price'] * t['size']) for t in trades]
        
        # Run Monte Carlo simulation
        final_equities = []
        max_drawdowns = []
        
        for _ in range(iterations):
            # Shuffle returns to simulate different paths
            np.random.shuffle(returns)
            
            # Calculate equity curve
            equity = [self.initial_capital]
            for r in returns:
                equity.append(equity[-1] * (1 + r))
            
            # Calculate maximum drawdown
            peak = equity[0]
            max_dd = 0
            
            for val in equity:
                peak = max(peak, val)
                dd = (peak - val) / peak
                max_dd = max(max_dd, dd)
            
            final_equities.append(equity[-1])
            max_drawdowns.append(max_dd)
        
        # Calculate statistics
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            'worst_drawdown': np.percentile(max_drawdowns, 95),
            'var_95': self.initial_capital - np.percentile(final_equities, 5),
            'expected_return': np.mean(final_equities) / self.initial_capital - 1
        }
    
    def _analyze_trade_stats(self, trades: List[Dict]) -> Dict:
        """Analyze trade statistics for insights"""
        if not trades:
            return {}
        
        # Basic stats
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        # Calculate various metrics
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        avg_win_duration = np.mean([(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in wins]) if wins else 0
        avg_loss_duration = np.mean([(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in losses]) if losses else 0
        
        # Return comprehensive stats
        return {
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_duration_hours': avg_win_duration,
            'avg_loss_duration_hours': avg_loss_duration,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'expectancy': (len(wins) / len(trades) * avg_win - len(losses) / len(trades) * abs(avg_loss)) if trades else 0
        }
    
    def _calculate_robustness_score(self, metrics: Dict, monte_carlo: Dict) -> float:
        """Calculate overall robustness score (0-100)"""
        # Base score on key metrics
        score = 0
        
        # Reward positive returns (0-30 points)
        if metrics.get('total_return', 0) > 0:
            score += min(30, metrics['total_return'] * 100)
        
        # Reward consistent returns (0-20 points)
        if 'r_squared' in metrics:
            score += metrics['r_squared'] * 20
        
        # Reward risk-adjusted returns (0-20 points)
        sharpe = metrics.get('sharpe_ratio', 0)
        score += min(20, max(0, sharpe * 10))
        
        # Reward drawdown management (0-15 points)
        max_dd = metrics.get('max_drawdown', 1)
        score += max(0, 15 * (1 - max_dd / 0.2))  # 15 points for drawdown below 20%
        
        # Reward Monte Carlo stability (0-15 points)
        mc_dd_ratio = metrics.get('max_drawdown', 0.01) / monte_carlo.get('worst_drawdown', 0.01)
        score += 15 * min(1, max(0, mc_dd_ratio))
        
        return min(100, max(0, score))