"""
Tests for data streaming functionality.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from src.data.stream import KrakenStreamManager

class TestStreamManager(unittest.TestCase):
    @patch('websockets.connect', new_callable=AsyncMock)
    async def test_connect(self, mock_connect):
        """Test WebSocket connection"""
        # Setup mock
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        mock_ws.recv.return_value = '{"event":"systemStatus","status":"online"}'
        
        # Create manager
        manager = KrakenStreamManager()
        
        # Test connect
        result = await manager.connect()
        
        # Verify
        self.assertTrue(result)
        mock_connect.assert_called_once()
        
    @patch('websockets.connect', new_callable=AsyncMock)
    async def test_subscribe(self, mock_connect):
        """Test subscribing to channels"""
        # Setup mock
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        mock_ws.recv.side_effect = [
            '{"event":"systemStatus","status":"online"}',
            '{"event":"subscriptionStatus","pair":"BTC/USD","status":"subscribed","channelName":"ticker"}'
        ]
        
        # Create manager
        manager = KrakenStreamManager()
        await manager.connect()
        
        # Test subscribe
        await manager.subscribe(['BTC/USD'], ['ticker'])
        
        # Verify
        self.assertEqual(mock_ws.send.call_count, 1)
        
    @patch('websockets.connect', new_callable=AsyncMock)
    async def test_handle_message(self, mock_connect):
        """Test message handling"""
        # Setup mock
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        
        # Create manager
        manager = KrakenStreamManager()
        
        # Setup callback
        callback_called = False
        callback_data = None
        
        def test_callback(symbol, data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        # Register callback
        manager.add_callback('ticker', test_callback)
        
        # Test message handling
        await manager._handle_message('[{"a":["5525.40000","1","1.000"]},"ticker","BTC/USD"]')
        
        # Verify callback was called
        self.assertTrue(callback_called)
        self.assertIsNotNone(callback_data)

    def test_stream_manager(self):
        """Run async tests"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.test_connect())
        loop.run_until_complete(self.test_subscribe())
        loop.run_until_complete(self.test_handle_message())