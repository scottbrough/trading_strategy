"""
Enhanced optimization module combining walk-forward, genetic algorithms, and Bayesian approaches.
Provides unified interfaces for parameter optimization with robustness testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from deap import base, creator, tools, algorithms
import multiprocessing
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import random
import optuna
from functools import partial
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from ..core.logger import log_manager
from ..core.config import config
from ..core.exceptions import StrategyError
from .backtest import BacktestEngine

logger = log_manager.get_logger(__name__)

class WalkForwardOptimizer:
    """
    Unified optimization framework combining multiple approaches.
    
    Features:
    - Walk-forward optimization to prevent overfitting
    - Genetic algorithm for complex parameter spaces
    - Bayesian optimization for efficient parameter search
    - Monte Carlo simulation for robustness testing
    - Cross-validation across market regimes
    """
    
    def __init__(self, 
                 strategy_class: type, 
                 data: Dict[str, pd.DataFrame],
                 parameter_ranges: Dict[str, Tuple[float, float]],
                 base_config: Dict[str, Any] = None,
                 metrics_weights: Dict[str, float] = None,
                 output_dir: str = "optimization_results"):
        """
        Initialize optimizer with strategy and data.
        
        Args:
            strategy_class: Class of the strategy to optimize
            data: Dictionary of DataFrames with historical data
            parameter_ranges: Dictionary of parameter ranges to test
            base_config: Base configuration for the strategy
            metrics_weights: Weights for different metrics in fitness calculation
            output_dir: Directory to save optimization results
        """
        self.strategy_class = strategy_class
        self.data = data
        self.parameter_ranges = parameter_ranges
        self.base_config = base_config or {}
        self.backtest_engine = BacktestEngine()
        self.config = config.get_optimization_params()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Default metric weights if not provided
        self.metrics_weights = metrics_weights or {
            'total_return': 0.4,
            'sharpe_ratio': 0.2,
            'sortino_ratio': 0.1,
            'max_drawdown': -0.2,  # Negative weight as we want to minimize drawdown
            'win_rate': 0.1,
            'profit_factor': 0.1,
            'trade_count_penalty': -0.1  # Penalty for excessive trading
        }
        
        # Initialize optimization structures
        self._init_deap()
        
        # Results storage
        self.results = {
            'genetic': [],
            'walk_forward': [],
            'bayesian': [],
            'best_parameters': None,
            'best_metrics': None
        }
        
        logger.info(f"Optimizer initialized with {len(parameter_ranges)} parameters to optimize")

    def _init_deap(self):
        """Initialize DEAP (genetic algorithm) components"""
        try:
            # Clear any existing DEAP classes to avoid warnings
            if 'FitnessMax' in creator.__dict__:
                del creator.FitnessMax
            if 'Individual' in creator.__dict__:
                del creator.Individual
            
            # Create fitness and individual classes
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Set up toolbox
            self.toolbox = base.Toolbox()
            
            # Register genetic operators
            self.toolbox.register("attr_float", self._generate_random_param)
            self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                                 self._create_individual)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("evaluate", self._evaluate_individual)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", self._adaptive_mutation)
            self.toolbox.register("select", tools.selTournament, tournsize=3)
            
        except Exception as e:
            logger.error(f"Error setting up genetic algorithm: {str(e)}")
            raise StrategyError.OptimizationError("Failed to setup genetic algorithm")
            
    def _generate_random_param(self):
        """Generate a random parameter value"""
        param_name = random.choice(list(self.parameter_ranges.keys()))
        param_range = self.parameter_ranges[param_name]
        
        if isinstance(param_range[0], int) and isinstance(param_range[1], int):
            return random.randint(param_range[0], param_range[1])
        elif isinstance(param_range[0], bool) or isinstance(param_range[1], bool):
            return random.choice([True, False])
        else:
            return random.uniform(param_range[0], param_range[1])
            
    def _create_individual(self):
        """Create a complete individual with all parameters"""
        params = {}
        
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            # Handle different parameter types
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                params[param_name] = random.randint(min_val, max_val)
            elif isinstance(min_val, bool) or isinstance(max_val, bool):
                # Boolean parameter
                params[param_name] = random.choice([True, False])
            else:
                # Float parameter
                params[param_name] = random.uniform(min_val, max_val)
        
        # Convert to list in a fixed order for genetic algorithm
        self.param_names = list(self.parameter_ranges.keys())
        return [params[name] for name in self.param_names]
    
    def _adaptive_mutation(self, individual, indpb=0.2, scale=0.2):
        """
        Adaptive mutation based on parameter type and range.
        Allows larger mutations early in optimization and finer adjustments later.
        """
        for i, param_name in enumerate(self.param_names):
            if random.random() < indpb:
                param_range = self.parameter_ranges[param_name]
                
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    # Integer parameter
                    range_size = param_range[1] - param_range[0]
                    mutation_size = max(1, int(range_size * scale * random.random()))
                    individual[i] += random.choice([-1, 1]) * mutation_size
                    individual[i] = max(param_range[0], min(param_range[1], individual[i]))
                
                elif isinstance(param_range[0], bool) or isinstance(param_range[1], bool):
                    # Boolean parameter - just flip it
                    individual[i] = not individual[i]
                
                else:
                    # Float parameter
                    range_size = param_range[1] - param_range[0]
                    mutation_size = range_size * scale * random.random()
                    individual[i] += random.choice([-1, 1]) * mutation_size
                    individual[i] = max(param_range[0], min(param_range[1], individual[i]))
        
        return individual,
    
    def _evaluate_individual(self, individual):
        """Evaluate an individual by running backtest"""
        try:
            # Convert individual to parameter dictionary
            params = {self.param_names[i]: individual[i] for i in range(len(individual))}
            
            # Create full configuration
            full_config = self.base_config.copy()
            strategy_params = full_config.get('strategy', {}).copy()
            strategy_params.update(params)
            full_config['strategy'] = strategy_params
            
            # Create strategy instance
            strategy = self.strategy_class(full_config)
            
            # Run backtest
            results = self.backtest_engine.run_backtest(
                strategy,
                self.data,
                transaction_costs=True
            )
            
            # Calculate combined fitness score
            fitness = self._calculate_fitness(results['metrics'])
            return (fitness,)
            
        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            return (-999,)  # Very low fitness score for failed evaluations
    
    def _calculate_fitness(self, metrics):
        """
        Calculate comprehensive fitness score from backtest metrics
        with configurable weights.
        """
        try:
            # Extract key metrics (with safety defaults)
            total_return = metrics.get('total_return', 0) 
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            sortino_ratio = metrics.get('sortino_ratio', 0)
            max_drawdown = metrics.get('max_drawdown', 1)
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            num_trades = metrics.get('total_trades', 0)
            
            # Normalize and scale metrics appropriately
            if max_drawdown > 0:
                drawdown_factor = np.exp(-5 * max_drawdown)
            else:
                drawdown_factor = 1.0
                
            # Penalty for too few or too many trades
            ideal_trade_count = 30  # Can be adjusted based on timeframe
            trade_count_penalty = -abs(num_trades - ideal_trade_count) / ideal_trade_count
            
            # Ensure profit factor is not infinite
            if profit_factor == float('inf'):
                profit_factor = 100
                
            # Combine metrics with weights
            fitness = (
                self.metrics_weights.get('total_return', 0.4) * total_return +
                self.metrics_weights.get('sharpe_ratio', 0.2) * sharpe_ratio +
                self.metrics_weights.get('sortino_ratio', 0.1) * sortino_ratio +
                self.metrics_weights.get('max_drawdown', -0.2) * drawdown_factor +
                self.metrics_weights.get('win_rate', 0.1) * win_rate +
                self.metrics_weights.get('profit_factor', 0.1) * min(profit_factor, 100) / 100 +
                self.metrics_weights.get('trade_count_penalty', -0.1) * trade_count_penalty
            )
            
            # Apply normalization and clamping
            fitness = max(0.001, min(fitness, 100))  # Ensure positive finite fitness
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error calculating fitness: {str(e)}")
            return 0.001  # Return minimal fitness value
    
    def optimize_genetic(self, population_size=50, generations=30, 
                         checkpoint_freq=5, continue_from=None):
        """
        Run genetic algorithm optimization
        
        Args:
            population_size: Size of population in each generation
            generations: Number of generations to evolve
            checkpoint_freq: How often to save checkpoints
            continue_from: Checkpoint file to continue from
            
        Returns:
            Dict with optimization results and best parameters
        """
        try:
            logger.info(f"Starting genetic algorithm optimization with population {population_size}, "
                       f"generations {generations}")
            
            # Initialize or load population
            if continue_from and os.path.exists(continue_from):
                population, logbook, gen = self._load_checkpoint(continue_from)
                start_gen = gen + 1
            else:
                population = self.toolbox.population(n=population_size)
                logbook = tools.Logbook()
                start_gen = 0
            
            # Statistics setup
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Hall of Fame
            hof = tools.HallOfFame(5)
            
            # Run evolution
            for gen in range(start_gen, generations):
                # Select the next generation individuals
                offspring = algorithms.varAnd(
                    population, self.toolbox,
                    cxpb=0.7,  # crossover probability
                    mutpb=0.3   # mutation probability
                )
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Update population and hall of fame
                population = self.toolbox.select(offspring, len(population))
                hof.update(population)
                
                # Record statistics
                record = stats.compile(population)
                logbook.record(gen=gen, **record)
                
                # Save checkpoint
                if gen % checkpoint_freq == 0:
                    self._save_checkpoint(gen, population, hof, logbook)
                
                # Log progress
                logger.info(f"Generation {gen}: {record}")
            
            # Get best parameters
            best_individual = hof[0]
            best_params = {self.param_names[i]: best_individual[i] 
                          for i in range(len(best_individual))}
            
            # Evaluate best individual
            strategy_config = self.base_config.copy()
            strategy_params = strategy_config.get('strategy', {}).copy()
            strategy_params.update(best_params)
            strategy_config['strategy'] = strategy_params
            
            strategy = self.strategy_class(strategy_config)
            results = self.backtest_engine.run_backtest(
                strategy,
                self.data,
                transaction_costs=True
            )
            
            # Store results
            genetic_result = {
                'best_parameters': best_params,
                'best_metrics': results['metrics'],
                'generations': generations,
                'population_size': population_size,
                'hall_of_fame': [list(ind) for ind in hof],
                'logbook': logbook
            }
            
            self.results['genetic'] = genetic_result
            
            # Update best overall parameters if this is better
            if (self.results['best_metrics'] is None or 
                results['metrics']['total_return'] > self.results['best_metrics']['total_return']):
                self.results['best_parameters'] = best_params
                self.results['best_metrics'] = results['metrics']
            
            return genetic_result
            
        except Exception as e:
            logger.error(f"Genetic optimization error: {str(e)}")
            raise StrategyError.OptimizationError("Failed to optimize parameters")
    
    def _save_checkpoint(self, generation, population, hof, logbook):
        """Save optimization checkpoint"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                'generation': generation,
                'population': [ind[:] for ind in population],
                'hof': [ind[:] for ind in hof],
                'logbook': dict(logbook),
                'param_names': self.param_names
            }
            
            checkpoint_file = self.output_dir / f"checkpoint_gen{generation}_{timestamp}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=4, default=lambda o: float(o) 
                          if isinstance(o, np.number) else o)
                
            logger.info(f"Saved checkpoint to {checkpoint_file}")
                
        except Exception as e:
            logger.error(f"Checkpoint save error: {str(e)}")
    
    def _load_checkpoint(self, filepath):
        """Load optimization checkpoint"""
        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore parameter names
            self.param_names = checkpoint.get('param_names', self.param_names)
            
            # Recreate population
            population = []
            for ind_list in checkpoint['population']:
                ind = creator.Individual(ind_list)
                population.append(ind)
            
            # Load generation number
            generation = checkpoint['generation']
            
            # Recreate logbook
            logbook = tools.Logbook()
            for entry in checkpoint['logbook'].get('chapters', []):
                logbook.record(**entry)
            
            logger.info(f"Loaded checkpoint from generation {generation}")
            return population, logbook, generation
            
        except Exception as e:
            logger.error(f"Checkpoint load error: {str(e)}")
            raise StrategyError.OptimizationError("Failed to load checkpoint")
    
    def optimize_walk_forward(self, num_folds=5, train_size=0.7, num_trials=20, 
                             random_trials=True):
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
            logger.info(f"Starting walk-forward optimization with {num_folds} folds")
            
            # Split data into folds
            folds = self._create_time_folds(num_folds, train_size)
            
            fold_results = []
            
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
                
                fold_results.append(fold_result)
                
                logger.info(f"Fold {fold_idx+1} - Best train metric: {best_metric:.4f}, "
                           f"Test return: {test_results['total_return']:.2%}")
            
            # Aggregate results
            best_combined_params = self._get_robust_parameters(fold_results)
            
            # Store in results
            walk_forward_result = {
                'best_parameters': best_combined_params,
                'fold_results': fold_results,
                'num_folds': num_folds,
                'train_size': train_size,
                'num_trials': num_trials
            }
            
            self.results['walk_forward'] = walk_forward_result
            
            # Test the combined parameters
            full_config = self.base_config.copy()
            strategy_params = full_config.get('strategy', {}).copy()
            strategy_params.update(best_combined_params)
            full_config['strategy'] = strategy_params
            
            strategy = self.strategy_class(full_config)
            results = self.backtest_engine.run_backtest(
                strategy,
                self.data,
                transaction_costs=True
            )
            
            # Update best overall if this is better
            if (self.results['best_metrics'] is None or 
                results['metrics']['total_return'] > self.results['best_metrics']['total_return']):
                self.results['best_parameters'] = best_combined_params
                self.results['best_metrics'] = results['metrics']
            
            logger.info("Walk-forward optimization completed")
            
            return walk_forward_result
            
        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {str(e)}")
            raise StrategyError.OptimizationError("Failed to run walk-forward optimization")
    
    def _create_time_folds(self, num_folds, train_size):
        """
        Create time-based folds with dynamic market regime detection
        """
        folds = []
        
        # Determine market regimes
        regime_data = self._detect_market_regimes()
        
        # Create TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=num_folds)
        
        # Get sorted list of all dates across all symbols
        all_dates = []
        for symbol, df in self.data.items():
            all_dates.extend(df.index.tolist())
        all_dates = sorted(list(set(all_dates)))
        
        # Create index array for splitting
        idx_array = np.arange(len(all_dates))
        
        # Create folds using TimeSeriesSplit
        for train_idx, test_idx in tscv.split(idx_array):
            # Calculate split point based on train_size
            split_point = int(len(train_idx) * train_size)
            train_dates = [all_dates[i] for i in train_idx[:split_point]]
            test_dates = [all_dates[i] for i in train_idx[split_point:]] + [all_dates[i] for i in test_idx]
            
            # Get min and max dates for train and test
            min_train_date = min(train_dates) if train_dates else None
            max_train_date = max(train_dates) if train_dates else None
            min_test_date = min(test_dates) if test_dates else None  
            max_test_date = max(test_dates) if test_dates else None
            
            if min_train_date is None or max_train_date is None or min_test_date is None or max_test_date is None:
                continue
                
            # Split data
            train_data = {}
            test_data = {}
            
            for symbol, df in self.data.items():
                # Get data for the date ranges
                train_df = df[(df.index >= min_train_date) & (df.index <= max_train_date)].copy()
                test_df = df[(df.index >= min_test_date) & (df.index <= max_test_date)].copy()
                
                if not train_df.empty:
                    train_data[symbol] = train_df
                if not test_df.empty:
                    test_data[symbol] = test_df
            
            if train_data and test_data:
                folds.append((train_data, test_data))
        
        return folds
    
    def _detect_market_regimes(self):
        """
        Detect different market regimes in the data for better fold creation
        """
        regime_data = {}
        
        for symbol, df in self.data.items():
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Create a Series for regimes with the same index as the volatility Series
            regimes = pd.Series(index=volatility.index, data='neutral')
            
            # Define regimes based on volatility
            high_vol_mask = volatility > volatility.quantile(0.7)
            low_vol_mask = volatility <= volatility.quantile(0.3)
            
            # Apply masks to the regimes Series
            regimes.loc[high_vol_mask] = 'volatile'
            regimes.loc[low_vol_mask] = 'low_vol'
            
            # Identify trends using simple method
            price_sma = df['close'].rolling(window=50).mean()
            price_sma = price_sma.loc[regimes.index]  # Align indices
            
            # Calculate price ratio only for dates in the regimes index
            df_aligned = df.loc[regimes.index]
            price_ratio = df_aligned['close'] / price_sma
            
            # Create trend masks
            uptrend_mask = (price_ratio > 1.02) & (regimes == 'neutral')
            downtrend_mask = (price_ratio < 0.98) & (regimes == 'neutral')
            
            # Apply trend masks
            regimes.loc[uptrend_mask] = 'uptrend'
            regimes.loc[downtrend_mask] = 'downtrend'
            
            regime_data[symbol] = regimes
            
        return regime_data
    
    def _generate_random_parameters(self, num_trials):
        """Generate random parameter combinations"""
        param_sets = []
        
        for _ in range(num_trials):
            params = {}
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
    
    def _generate_grid_parameters(self, max_trials):
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
        
        # Generate combinations (up to max_trials)
        combinations = []
        self._recursive_grid_generation(param_values, {}, list(param_values.keys()), 0, combinations, max_trials)
        
        return combinations[:max_trials]
    
    def _recursive_grid_generation(self, param_values, current_params, param_names, idx, result, max_count):
        """Helper function for recursive grid generation"""
        if idx == len(param_names) or len(result) >= max_count:
            result.append(current_params.copy())
            return
        
        param_name = param_names[idx]
        for value in param_values[param_name]:
            current_params[param_name] = value
            self._recursive_grid_generation(param_values, current_params, param_names, idx+1, result, max_count)
            if len(result) >= max_count:
                return
    
    def _evaluate_parameters(self, parameter_sets, data_dict):
        """Evaluate multiple parameter sets on data"""
        results = []
        
        for params in parameter_sets:
            try:
                # Create full configuration
                full_config = self.base_config.copy()
                strategy_params = full_config.get('strategy', {}).copy()
                strategy_params.update(params)
                full_config['strategy'] = strategy_params
                
                # Create strategy
                strategy = self.strategy_class(full_config)
                
                # Run backtest
                backtest_result = self.backtest_engine.run_backtest(
                    strategy,
                    data_dict,
                    transaction_costs=True
                )
                
                # Calculate fitness
                metrics = backtest_result['metrics']
                fitness = self._calculate_fitness(metrics)
                
                # Store results
                result = {
                    'parameters': params,
                    'metrics': metrics,
                    'fitness': fitness
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating parameters: {str(e)}")
                continue
        
        return results
    
    def _get_best_parameters(self, results):
        """Get best parameters based on fitness score"""
        if not results:
            return {}, 0.0
            
        # Sort by fitness (higher is better)
        sorted_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
        
        return sorted_results[0]['parameters'], sorted_results[0]['fitness']
    
    def _test_parameters(self, parameters, data_dict):
        """Test parameters on out-of-sample data"""
        try:
            # Create full configuration
            full_config = self.base_config.copy()
            strategy_params = full_config.get('strategy', {}).copy()
            strategy_params.update(parameters)
            full_config['strategy'] = strategy_params
            
            # Create strategy
            strategy = self.strategy_class(full_config)
            
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
    
    def _get_robust_parameters(self, fold_results):
        """Get robust parameters by weighting across folds"""
        if not fold_results:
            return {}
            
        # Initialize parameter accumulator
        combined_params = {}
        weights = {}
        total_weight = 0
        
        for fold_result in fold_results:
            # Use test return as weight (higher return = higher weight)
            # Add penalty for negative returns
            test_return = fold_result['test_metrics'].get('total_return', 0)
            weight = max(0, test_return + 0.2)  # Add 0.2 to avoid very low weights
            total_weight += weight
            
            # Add weighted parameters
            for param_name, param_value in fold_result['best_parameters'].items():
                if param_name not in combined_params:
                    combined_params[param_name] = 0
                    weights[param_name] = 0
                    
                combined_params[param_name] += param_value * weight
                weights[param_name] += weight
        
        # Normalize by total weight
        result_params = {}
        for param_name, weighted_value in combined_params.items():
            param_weight = weights.get(param_name, 0)
            if param_weight > 0:
                # Get the original parameter range to determine type
                param_range = self.parameter_ranges.get(param_name, (0, 1))
                is_int = isinstance(param_range[0], int) and isinstance(param_range[1], int)
                is_bool = isinstance(param_range[0], bool) or isinstance(param_range[1], bool)
                
                # Normalize and convert to the appropriate type
                normalized_value = weighted_value / param_weight
                
                if is_int:
                    result_params[param_name] = int(round(normalized_value))
                elif is_bool:
                    result_params[param_name] = normalized_value >= 0.5
                else:
                    result_params[param_name] = normalized_value
        
        return result_params
    
    def optimize_bayesian(self, n_trials=100, timeout=3600):
        """
        Run Bayesian optimization using Optuna
        
        Args:
            n_trials: Maximum number of trials
            timeout: Timeout in seconds
            
        Returns:
            Dict with optimization results
        """
        try:
            def objective(trial):
                # Generate parameters using Optuna
                params = {}
                for param_name, (min_val, max_val) in self.parameter_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    elif isinstance(min_val, bool) or isinstance(max_val, bool):
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])
                    else:
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                
                # Create full configuration
                full_config = self.base_config.copy()
                strategy_params = full_config.get('strategy', {}).copy()
                strategy_params.update(params)
                full_config['strategy'] = strategy_params
                
                # Create strategy
                strategy = self.strategy_class(full_config)
                
                # Run backtest
                backtest_result = self.backtest_engine.run_backtest(
                    strategy,
                    self.data,
                    transaction_costs=True
                )
                
                # Calculate fitness
                fitness = self._calculate_fitness(backtest_result['metrics'])
                
                # Store trial details
                trial.set_user_attr('metrics', backtest_result['metrics'])
                
                return fitness
            
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Get best parameters
            best_params = study.best_params
            
            # Get metrics of best trial
            best_metrics = study.best_trial.user_attrs['metrics']
            
            # Store results
            bayesian_result = {
                'best_parameters': best_params,
                'best_metrics': best_metrics,
                'n_trials': n_trials,
                'study': study
            }
            
            self.results['bayesian'] = bayesian_result
            
            # Update best overall if better
            if (self.results['best_metrics'] is None or 
                best_metrics['total_return'] > self.results['best_metrics']['total_return']):
                self.results['best_parameters'] = best_params
                self.results['best_metrics'] = best_metrics
            
            return bayesian_result
            
        except Exception as e:
            logger.error(f"Bayesian optimization error: {str(e)}")
            raise StrategyError.OptimizationError("Failed to run Bayesian optimization")
    
    def run_monte_carlo(self, parameters=None, num_simulations=1000):
        """
        Run Monte Carlo simulation to analyze strategy robustness
        
        Args:
            parameters: Parameters to use (uses best parameters if None)
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dict with Monte Carlo analysis results
        """
        try:
            # Use provided parameters or best parameters
            params = parameters or self.results['best_parameters']
            if not params:
                raise ValueError("No parameters available for Monte Carlo analysis")
            
            # Create strategy with these parameters
            full_config = self.base_config.copy()
            strategy_params = full_config.get('strategy', {}).copy()
            strategy_params.update(params)
            full_config['strategy'] = strategy_params
            
            strategy = self.strategy_class(full_config)
            
            # Run backtest to get trades
            backtest_result = self.backtest_engine.run_backtest(
                strategy,
                self.data,
                transaction_costs=True
            )
            
            trades = backtest_result['trades']
            
            if not trades:
                raise ValueError("No trades to analyze")
            
            # Extract P&L data from trades
            pnl_values = [trade['pnl'] for trade in trades]
            pnl_pct_values = [(trade['exit_price'] / trade['entry_price'] - 1) * 100 
                             if trade['side'] == 'long'
                             else (trade['entry_price'] / trade['exit_price'] - 1) * 100
                             for trade in trades]
            
            # Run Monte Carlo simulations
            initial_capital = 10000  # default initial capital
            equity_curves = []
            max_drawdowns = []
            final_returns = []
            
            for _ in range(num_simulations):
                # Shuffle returns to simulate different order of trades
                np.random.shuffle(pnl_pct_values)
                
                # Calculate equity curve
                equity = [initial_capital]
                
                for r in pnl_pct_values:
                    # Apply return to previous equity
                    next_equity = equity[-1] * (1 + r/100)
                    equity.append(next_equity)
                
                # Calculate maximum drawdown
                peak = equity[0]
                max_dd = 0
                
                for val in equity:
                    peak = max(peak, val)
                    dd = (peak - val) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                
                equity_curves.append(equity)
                max_drawdowns.append(max_dd)
                final_returns.append(equity[-1] / equity[0] - 1)
            
            # Calculate statistics
            mc_result = {
                'final_equity_mean': np.mean(final_returns),
                'final_equity_median': np.median(final_returns),
                'final_equity_std': np.std(final_returns),
                'max_drawdown_mean': np.mean(max_drawdowns),
                'max_drawdown_median': np.median(max_drawdowns),
                'max_drawdown_95th': np.percentile(max_drawdowns, 95),
                'var_95': np.percentile(final_returns, 5),  # 5th percentile as 95% VaR
                'profitable_ratio': np.mean([1 if r > 0 else 0 for r in final_returns]),
                'equity_curves': equity_curves,
                'parameters': params
            }
            
            return mc_result
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {str(e)}")
            raise StrategyError.OptimizationError("Failed to run Monte Carlo simulation")
    
    def save_results(self, filename=None):
        """Save optimization results to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.output_dir / f"optimization_result_{timestamp}.json"
            
            # Prepare results for serialization
            serialized_results = {
                'best_parameters': self.results['best_parameters'],
                'best_metrics': self.results['best_metrics'],
                'parameter_ranges': self.parameter_ranges,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add method-specific results if available
            for method in ['genetic', 'walk_forward', 'bayesian']:
                if self.results[method]:
                    method_result = self.results[method].copy()
                    
                    # Remove non-serializable objects
                    if method == 'genetic':
                        if 'logbook' in method_result:
                            method_result['logbook'] = str(method_result['logbook'])
                    elif method == 'bayesian':
                        if 'study' in method_result:
                            del method_result['study']
                    
                    serialized_results[method] = method_result
            
            with open(filename, 'w') as f:
                json.dump(serialized_results, f, indent=4, default=lambda o: float(o) 
                          if isinstance(o, np.number) else str(o))
                
            logger.info(f"Optimization results saved to {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return None
    
    def plot_results(self, show_plots=True, save_dir=None):
        """
        Plot optimization results
        
        Args:
            show_plots: Whether to display plots
            save_dir: Directory to save plots (None to not save)
        """
        try:
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(exist_ok=True, parents=True)
            
            # Plot method comparison if we have multiple methods
            methods_used = sum(1 for m in ['genetic', 'walk_forward', 'bayesian'] 
                              if self.results[m])
            
            if methods_used > 1:
                self._plot_method_comparison(show_plots, save_dir)
            
            # Genetic algorithm convergence plot
            if self.results['genetic']:
                self._plot_genetic_convergence(show_plots, save_dir)
            
            # Walk-forward performance plot
            if self.results['walk_forward']:
                self._plot_walkforward_performance(show_plots, save_dir)
            
            # Bayesian optimization plot
            if self.results['bayesian']:
                self._plot_bayesian_results(show_plots, save_dir)
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
    
    def _plot_method_comparison(self, show_plots, save_dir):
        """Plot comparison of optimization methods"""
        try:
            methods = []
            returns = []
            sharpes = []
            drawdowns = []
            
            if self.results['genetic']:
                methods.append('Genetic')
                returns.append(self.results['genetic']['best_metrics']['total_return'])
                sharpes.append(self.results['genetic']['best_metrics']['sharpe_ratio'])
                drawdowns.append(self.results['genetic']['best_metrics']['max_drawdown'])
                
            if self.results['walk_forward']:
                methods.append('Walk-Forward')
                returns.append(self.results['walk_forward']['best_metrics']['total_return'])
                sharpes.append(self.results['walk_forward']['best_metrics']['sharpe_ratio'])
                drawdowns.append(self.results['walk_forward']['best_metrics']['max_drawdown'])
                
            if self.results['bayesian']:
                methods.append('Bayesian')
                returns.append(self.results['bayesian']['best_metrics']['total_return'])
                sharpes.append(self.results['bayesian']['best_metrics']['sharpe_ratio'])
                drawdowns.append(self.results['bayesian']['best_metrics']['max_drawdown'])
            
            if not methods:
                return
                
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Returns
            ax1.bar(methods, returns)
            ax1.set_title('Total Return Comparison')
            ax1.set_ylabel('Return')
            
            # Sharpe
            ax2.bar(methods, sharpes)
            ax2.set_title('Sharpe Ratio Comparison')
            ax2.set_ylabel('Sharpe Ratio')
            
            # Drawdown
            ax3.bar(methods, drawdowns)
            ax3.set_title('Max Drawdown Comparison')
            ax3.set_ylabel('Max Drawdown')
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(Path(save_dir) / 'method_comparison.png')
                
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting method comparison: {str(e)}")
    
    def _plot_genetic_convergence(self, show_plots, save_dir):
        """Plot genetic algorithm convergence"""
        try:
            genetic_result = self.results['genetic']
            if 'logbook' not in genetic_result:
                return
                
            # Extract data from logbook
            gen = range(len(genetic_result['logbook']))
            avg = genetic_result['logbook'].select('avg')
            max_fit = genetic_result['logbook'].select('max')
            
            plt.figure(figsize=(10, 6))
            plt.plot(gen, avg, 'k-', label='Average Fitness')
            plt.plot(gen, max_fit, 'r-', label='Best Fitness')
            plt.title('Genetic Algorithm Convergence')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                plt.savefig(Path(save_dir) / 'genetic_convergence.png')
                
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting genetic convergence: {str(e)}")
    
    def _plot_walkforward_performance(self, show_plots, save_dir):
        """Plot walk-forward performance by fold"""
        try:
            wf_result = self.results['walk_forward']
            if 'fold_results' not in wf_result:
                return
                
            fold_results = wf_result['fold_results']
            folds = [r['fold'] for r in fold_results]
            train_metrics = [r['train_metric'] for r in fold_results]
            test_returns = [r['test_metrics']['total_return'] for r in fold_results]
            
            plt.figure(figsize=(10, 6))
            plt.bar(folds, train_metrics, alpha=0.7, label='Train Metric')
            plt.bar(folds, test_returns, alpha=0.7, label='Test Return')
            plt.title('Walk-Forward Performance by Fold')
            plt.xlabel('Fold')
            plt.ylabel('Performance Metric')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                plt.savefig(Path(save_dir) / 'walkforward_performance.png')
                
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting walk-forward performance: {str(e)}")
    
    def _plot_bayesian_results(self, show_plots, save_dir):
        """Plot Bayesian optimization results"""
        try:
            # This requires access to the optuna study object
            # If we don't have it, we'll skip the plot
            bayesian_result = self.results['bayesian']
            if 'study' not in bayesian_result:
                return
                
            # Plot optimization history
            study = bayesian_result['study']
            
            plt.figure(figsize=(10, 6))
            
            # Optimization history
            trials = range(len(study.trials))
            values = [t.value for t in study.trials]
            
            plt.plot(trials, values, 'o-')
            plt.axhline(y=study.best_value, color='r', linestyle='-', label=f'Best Value: {study.best_value:.4f}')
            plt.title('Bayesian Optimization History')
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                plt.savefig(Path(save_dir) / 'bayesian_optimization.png')
                
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting Bayesian results: {str(e)}")