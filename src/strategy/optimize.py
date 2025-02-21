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
        """Calculate fitness score from backtest metrics"""
        try:
            # Multi-objective fitness calculation
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            max_drawdown = metrics.get('max_drawdown', 1)
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            
            # Penalize excessive drawdown
            drawdown_penalty = np.exp(-max_drawdown * 10)
            
            # Combine metrics with weights
            fitness = (
                0.4 * sharpe_ratio +
                0.3 * drawdown_penalty +
                0.2 * win_rate +
                0.1 * profit_factor
            )
            
            return max(fitness, 0)  # Ensure non-negative fitness
            
        except Exception as e:
            logger.error(f"Fitness calculation error: {str(e)}")
            return 0.0
    
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
            raise StrategyError.OptimizationError("Failed to load checkpoint")