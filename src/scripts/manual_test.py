#!/usr/bin/env python
"""
Script to manually test the trading system by generating a few trades.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import the modules
from src.core.logger import log_manager
from src.core.config import config
from src.trading.paper_trading import PaperTradingExecutor
from src.data.database import db

logger = log_manager.get_logger(__name__)

async def run_manual_test():
    """Run a manual test of the trading system"""
    try:
        logger.info("Starting manual test...")
        
        # Initialize paper trading executor
        executor = PaperTradingExecutor()
        
        # Generate mock price updates
        executor.generate_mock_price_updates()
        
        # Wait for initial price updates
        logger.info("Waiting for initial price updates...")
        await asyncio.sleep(5)
        
        # Get current prices
        btc_ticker = await executor.get_ticker('BTC/USD')
        eth_ticker = await executor.get_ticker('ETH/USD')
        
        btc_price = btc_ticker['last_price']
        eth_price = eth_ticker['last_price']
        
        logger.info(f"Current BTC price: ${btc_price:.2f}")
        logger.info(f"Current ETH price: ${eth_price:.2f}")
        
        # Execute some test trades
        # 1. Buy BTC
        logger.info("Executing test trade 1: Buy BTC")
        signal1 = {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USD',
            'action': 'buy',
            'price': btc_price,
            'size': 0.1
        }
        await executor.add_signal(signal1)
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # 2. Buy ETH
        logger.info("Executing test trade 2: Buy ETH")
        signal2 = {
            'timestamp': datetime.now(),
            'symbol': 'ETH/USD',
            'action': 'buy',
            'price': eth_price,
            'size': 1.0
        }
        await executor.add_signal(signal2)
        
        # Wait for price updates
        logger.info("Waiting for price updates...")
        await asyncio.sleep(15)
        
        # 3. Sell half BTC position
        logger.info("Executing test trade 3: Sell half BTC position")
        signal3 = {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USD',
            'action': 'sell',
            'price': btc_ticker['last_price'] * 1.01,  # Slightly higher price
            'size': 0.05
        }
        await executor.add_signal(signal3)
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # 4. Exit ETH position
        logger.info("Executing test trade 4: Exit ETH position")
        signal4 = {
            'timestamp': datetime.now(),
            'symbol': 'ETH/USD',
            'action': 'exit',
            'price': eth_ticker['last_price'] * 1.02  # Higher price for profit
        }
        await executor.add_signal(signal4)
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Check account status
        balance = executor.get_account_balance()
        positions = executor.get_positions()
        trades = executor.get_trade_history()
        
        logger.info(f"Account balance: ${balance['balance']:.2f}")
        logger.info(f"Account equity: ${balance['equity']:.2f}")
        logger.info(f"Unrealized P&L: ${balance['unrealized_pnl']:.2f}")
        logger.info(f"Open positions: {len(positions)}")
        logger.info(f"Completed trades: {len(trades)}")
        
        # Store trades in database
        logger.info("Storing test trades in database...")
        for trade in trades:
            db_trade = {
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'amount': trade['size'],
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'pnl': trade['pnl'],
                'status': 'closed',
                'strategy': 'test_strategy'
            }
            db.store_trade(db_trade)
        
        logger.info(f"Manual test completed with {len(trades)} trades")
        
    except Exception as e:
        logger.error(f"Manual test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_manual_test())