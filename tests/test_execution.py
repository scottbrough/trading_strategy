"""
Tests for order execution system.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime
from src.trading.execution import OrderExecutor
from src.data.stream import KrakenStreamManager
from src.strategy.risk_management import RiskManager

class TestOrderExecutor(unittest.TestCase):
    def setUp(self):
        """Set up mocks and executor"""
        self.mock_stream = Mock(spec=KrakenStreamManager)
        self.mock_risk = Mock(spec=RiskManager)
        
        # Make stream methods async
        self.mock_stream.market_buy = AsyncMock()
        self.mock_stream.market_sell = AsyncMock()
        self.mock_stream.get_account_balance = AsyncMock(return_value=10000)
        
        # Setup risk manager
        self.mock_risk.check_risk_limits = Mock(return_value=True)
        self.mock_risk.calculate_position_size = Mock(return_value=1.0)
        
        # Create executor
        self.executor = OrderExecutor(self.mock_stream, self.mock_risk)
        
    async def test_add_signal(self):
        """Test adding a signal"""
        # Create signal
        signal = {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USD',
            'action': 'buy',
            'price': 50000.0
        }
        
        # Add signal
        result = await self.executor.add_signal(signal)
        
        # Verify
        self.assertTrue(result)
        self.assertEqual(len(self.executor.pending_signals), 1)
        
    async def test_process_buy_signal(self):
        """Test processing a buy signal"""
        # Create signal
        signal = {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USD',
            'action': 'buy',
            'price': 50000.0
        }
        
        # Setup mock response
        self.mock_stream.market_buy.return_value = {
            'order_id': 'test123',
            'status': 'filled',
            'filled_price': 50000.0,
            'filled_size': 1.0
        }
        
        # Process signal
        await self.executor._process_signal(signal)
        
        # Verify
        self.mock_risk.check_risk_limits.assert_called_once()
        self.mock_stream.market_buy.assert_called_once()
        self.assertEqual(len(self.executor.positions), 1)
        
    async def test_process_exit_signal(self):
        """Test processing an exit signal"""
        # Add position first
        self.executor.positions['BTC/USD'] = {
            'side': 'long',
            'size': 1.0,
            'entry_price': 50000.0,
            'entry_time': datetime.now(),
            'order_id': 'test123'
        }
        
        # Create exit signal
        signal = {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USD',
            'action': 'exit',
            'price': 55000.0
        }
        
        # Setup mock response
        self.mock_stream.market_sell.return_value = {
            'order_id': 'test456',
            'status': 'filled',
            'filled_price': 55000.0,
            'filled_size': 1.0
        }
        
        # Process signal
        await self.executor._process_signal(signal)
        
        # Verify
        self.mock_stream.market_sell.assert_called_once()
        self.assertEqual(len(self.executor.positions), 0)  # Position closed
        
    def test_executor(self):
        """Run async tests"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.test_add_signal())
        loop.run_until_complete(self.test_process_buy_signal())
        loop.run_until_complete(self.test_process_exit_signal())