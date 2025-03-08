# src/strategy/runner.py

"""
Strategy runner system for loading and running trading strategies.
"""

import importlib
import asyncio
import json
from typing import Dict, List, Any, Optional
import inspect
from datetime import datetime, timedelta
import time
import threading

from ..core.logger import log_manager
from ..core.config import config
from ..core.exceptions import StrategyError
from ..data.stream import KrakenStreamManager
from ..data.processor import DataProcessor
from ..trading.execution import OrderExecutor
from .risk_management import RiskManager

logger = log_manager.get_logger(__name__)

class StrategyRunner:
    def __init__(self, paper_trading=False):
        """Initialize strategy runner"""
        self.stream_manager = KrakenStreamManager()
        self.data_processor = DataProcessor()
        self.risk_manager = RiskManager(config.get('risk_params', {}))
        
        # Choose executor based on mode
        if paper_trading or config.is_sandbox():
            from ..trading.paper_trading import PaperTradingExecutor
            self.executor = PaperTradingExecutor()
            self.is_paper_trading = True
            logger.info("Using paper trading executor")
        else:
            self.executor = OrderExecutor(self.stream_manager, self.risk_manager)
            self.is_paper_trading = False
            logger.info("Using live order executor")
        
        self.strategies = {}
        self.running = False
        self.loop_thread = None
        
    async def initialize(self):
        """Initialize components"""
        try:
            # Connect to exchange
            await self.stream_manager.connect()
            
            # Start executor
            await self.executor.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy runner: {str(e)}")
            return False
            
    async def load_strategy(self, strategy_name: str, strategy_config: Dict[str, Any] = None, class_name: str = None) -> bool:
        """Load a strategy module and initialize the strategy"""
        try:
            # Import strategy module
            module_path = f"strategy.implementations.{strategy_name}"
            module = importlib.import_module(module_path)
            
            # Get the strategy class
            if class_name is None:
                # Try to find the class in the module
                class_name = strategy_name.split('.')[-1]
                # Convert snake_case to PascalCase
                class_name = ''.join(word.capitalize() for word in class_name.split('_'))
                if not class_name.endswith('Strategy'):
                    class_name = f"{class_name}Strategy"
            
            # Try to get the class
            if hasattr(module, class_name):
                strategy_class = getattr(module, class_name)
            else:
                # Search for any class that ends with 'Strategy'
                strategy_classes = [obj for name, obj in inspect.getmembers(module) 
                                if inspect.isclass(obj) and name.endswith('Strategy')]
                if strategy_classes:
                    strategy_class = strategy_classes[0]
                    class_name = strategy_classes[0].__name__
                    logger.info(f"Found strategy class: {class_name}")
                else:
                    raise AttributeError(f"No strategy class found in module {module_path}")
            
            # Initialize strategy with config
            strategy_instance = strategy_class(strategy_config or {})
            
            # Add to strategies
            self.strategies[strategy_name] = {
                'instance': strategy_instance,
                'config': strategy_config or {},
                'symbols': strategy_config.get('symbols', []),
                'timeframes': strategy_config.get('timeframes', ['1h']),
                'last_run': None
            }
            
            logger.info(f"Loaded strategy: {class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load strategy {strategy_name}: {str(e)}")
            return False
            
    async def start(self):
        """Start running strategies"""
        if self.running:
            logger.warning("Strategy runner already running")
            return False
            
        # Initialize components
        await self.initialize()
        
        # Start running
        self.running = True
        
        # Start in a separate thread
        self.loop_thread = threading.Thread(target=self._run_loop_wrapper)
        self.loop_thread.daemon = True
        self.loop_thread.start()
        
        logger.info("Strategy runner started")
        return True
        
    async def stop(self):
        """Stop running strategies"""
        self.running = False
        
        # Wait for thread to finish
        if self.loop_thread:
            self.loop_thread.join(timeout=5.0)
            
        # Stop executor
        await self.executor.stop()
        
        logger.info("Strategy runner stopped")
        
    def _run_loop_wrapper(self):
        """Wrapper to run asyncio loop in a thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._run_loop())
        except Exception as e:
            logger.error(f"Error in strategy run loop: {str(e)}")
        finally:
            loop.close()
            
    async def _run_loop(self):
        """Main strategy running loop"""
        while self.running:
            try:
                # Run each strategy
                for strategy_name, strategy_info in self.strategies.items():
                    await self._run_strategy(strategy_name, strategy_info)
                    
                # Sleep between runs
                await asyncio.sleep(10)  # 10 seconds between runs
                
            except Exception as e:
                logger.error(f"Error in strategy run loop: {str(e)}")
                await asyncio.sleep(10)  # Sleep on error
                
    async def _run_strategy(self, strategy_name: str, strategy_info: Dict[str, Any]):
        """Run a single strategy"""
        try:
            # Check if it's time to run this strategy
            current_time = datetime.now()
            last_run = strategy_info.get('last_run')
            
            # Minimum interval between runs
            min_interval = strategy_info['config'].get('run_interval', 60)  # seconds
            
            if last_run and (current_time - last_run).total_seconds() < min_interval:
                return  # Not time to run yet
                
            # Update last run time
            strategy_info['last_run'] = current_time
            
            # Get required data
            data_dict = {}
            for symbol in strategy_info['symbols']:
                for timeframe in strategy_info['timeframes']:
                    try:
                        # Get recent data
                        start_time = current_time - timedelta(days=30)  # 30 days should be enough
                        end_time = current_time
                        
                        df = self.stream_manager.get_ohlcv_data(symbol, timeframe, start_time, end_time)
                        
                        if df.empty:
                            logger.warning(f"No data for {symbol} {timeframe}")
                            continue
                            
                        # Process data
                        processed_df = self.data_processor.process_ohlcv(df, timeframe)
                        
                        key = f"{symbol}_{timeframe}"
                        data_dict[key] = processed_df
                        
                    except Exception as symbol_error:
                        logger.error(f"Error getting data for {symbol} {timeframe}: {str(symbol_error)}")
                        continue
                        
            if not data_dict:
                logger.warning(f"No data available for strategy {strategy_name}")
                return
                
            # Generate signals
            strategy_instance = strategy_info['instance']
            signals = []
            
            for key, df in data_dict.items():
                symbol = key.split('_')[0]
                
                try:
                    # Generate signals for this symbol
                    symbol_signals = strategy_instance.generate_signals(df)
                    
                    # Add symbol to signals if not present
                    for signal in symbol_signals:
                        if 'symbol' not in signal:
                            signal['symbol'] = symbol
                        signals.append(signal)
                        
                except Exception as signal_error:
                    logger.error(f"Error generating signals for {symbol}: {str(signal_error)}")
                    continue
                    
            if not signals:
                logger.info(f"No signals generated by strategy {strategy_name}")
                return
                
            # Filter recent signals only
            recent_signals = []
            for signal in signals:
                signal_time = signal.get('timestamp')
                
                # Only use signals from the last candle
                if isinstance(signal_time, datetime) and (current_time - signal_time).total_seconds() < 3600:
                    recent_signals.append(signal)
                    
            if not recent_signals:
                logger.info(f"No recent signals from strategy {strategy_name}")
                return
                
            # Execute signals
            for signal in recent_signals:
                await self.executor.add_signal(signal)
                
            logger.info(f"Strategy {strategy_name} generated {len(recent_signals)} signals")
            
        except Exception as e:
            logger.error(f"Error running strategy {strategy_name}: {str(e)}")