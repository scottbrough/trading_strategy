"""
Tests for configuration management.
"""

import unittest
import os
import tempfile
import yaml
from src.core.config import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        """Set up test config"""
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_dir = os.path.join(self.test_dir.name, 'config')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create test config files
        self.test_config = {
            'environment': 'test',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'test_db',
                'user': 'test_user',
                'password': 'test_password',
                'type': 'postgresql'
            },
            'exchange': {
                'name': 'kraken',
                'sandbox': True,
                'api_url': 'https://api.test.kraken.com',
                'websocket_url': 'wss://ws.test.kraken.com',
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'rate_limit': 0.2
            },
            'data': {
                'symbols': ['BTC/USD', 'ETH/USD'],
                'timeframes': ['1h', '4h']
            }
        }
        
        # Write config file
        with open(os.path.join(self.config_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.test_config, f)
            
        # Set config dir path for testing
        self.original_config_dir = ConfigManager.config_dir
        ConfigManager.config_dir = self.config_dir
        
    def tearDown(self):
        """Clean up"""
        self.test_dir.cleanup()
        ConfigManager.config_dir = self.original_config_dir
        
    def test_load_config(self):
        """Test loading configuration"""
        manager = ConfigManager()
        
        # Verify config loaded
        self.assertEqual(manager.config['environment'], 'test')
        self.assertEqual(manager.config['database']['name'], 'test_db')
        
    def test_get_methods(self):
        """Test config getter methods"""
        manager = ConfigManager()
        
        # Test various getters
        self.assertTrue(manager.is_sandbox())
        self.assertEqual(manager.get_symbols(), ['BTC/USD', 'ETH/USD'])
        self.assertEqual(manager.get_timeframes(), ['1h', '4h'])
        
    def test_get_by_path(self):
        """Test getting config by path"""
        manager = ConfigManager()
        
        # Test get method
        self.assertEqual(manager.get('database.name'), 'test_db')
        self.assertEqual(manager.get('exchange.api_key'), 'test_key')
        self.assertEqual(manager.get('nonexistent.key', 'default'), 'default')