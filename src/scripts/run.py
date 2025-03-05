#!/usr/bin/env python
"""
Main script for running the trading system.
Initializes and starts all components.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import the modules
from src.core.logger import log_manager
from src.core.config import config
from src.strategy.runner import StrategyRunner
from src.data.stream import KrakenStreamManager
from src.data.processor import DataProcessor
from src.strategy.risk_management import RiskManager
from src.trading.execution import OrderExecutor
from src.trading.paper_trading import PaperTradingExecutor

logger = log_manager.get_logger(__name__)

async def main():
    """Main function to start the trading system"""
    parser = argparse.ArgumentParser(description='Run trading system')
    parser.add_argument('--strategy', type=str, default='momentum_strategy',
                       help='Strategy to run (e.g., momentum_strategy)')
    parser.add_argument('--paper', action='store_true',
                       help='Use paper trading')
    args = parser.parse_args()
    
    try:
        logger.info("Starting trading system...")
        
        # Initialize components
        stream_manager = KrakenStreamManager()
        data_processor = DataProcessor()
        risk_manager = RiskManager(config.get_risk_params())
        
        # Initialize executor based on mode
        if args.paper or config.is_sandbox():
            logger.info("Using paper trading executor")
            executor = PaperTradingExecutor()
        else:
            logger.info("Using live trading executor")
            executor = OrderExecutor(stream_manager, risk_manager)
        
        # Initialize strategy runner
        runner = StrategyRunner()
        await runner.initialize()
        
        # Load strategy
        strategy_name = args.strategy
        logger.info(f"Loading strategy: {strategy_name}")
        strategy_config = {
            'symbols': config.get_symbols(),
            'timeframes': config.get_timeframes(),
            **config.get_trading_params()
        }

        # If strategy is provided as snake_case, convert to PascalCase for class name
        class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
        if not class_name.endswith('Strategy'):
            class_name = f"{class_name}Strategy"

        logger.info(f"Looking for strategy class: {class_name}")
        success = await runner.load_strategy(strategy_name, strategy_config, class_name)
        
        
        success = await runner.load_strategy(strategy_name, strategy_config)
        if not success:
            logger.error(f"Failed to load strategy: {strategy_name}")
            return
            
        # Start runner
        logger.info("Starting strategy runner...")
        await runner.start()
        
        # Keep the script running
        logger.info("Trading system running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Stopping trading system...")
        if 'runner' in locals():
            await runner.stop()
        logger.info("Trading system stopped")
    except Exception as e:
        logger.error(f"Error in trading system: {str(e)}")
        
if __name__ == "__main__":
    asyncio.run(main())