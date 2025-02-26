import unittest
import os
import tempfile
import yaml
from pathlib import Path
from src.core.config import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        """Set up test config"""
        # Save original config_dir
        self.original_config_dir = ConfigManager.config_dir
        
        # Create temporary directory for tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_config_dir = Path(self.test_dir.name) / 'config'
        self.test_config_dir.mkdir(exist_ok=True)
        
        # Create test config
        self.test_config = {
            'environment': 'sandbox',
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
        
        # Write config to file
        with open(self.test_config_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Set test config directory
        ConfigManager.config_dir = self.test_config_dir
        
        # Reset config manager to force reload
        ConfigManager.reset()
        self.config_manager = ConfigManager()
        self.config_manager.load_config()
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore original config directory
        ConfigManager.config_dir = self.original_config_dir
        ConfigManager.reset()
        
        # Clean up temp directory
        self.test_dir.cleanup()
    
    def test_load_config(self):
        """Test loading configuration"""
        # Verify config loaded correctly
        self.assertEqual(self.config_manager.config['environment'], 'sandbox')
        self.assertEqual(self.config_manager.config['database']['name'], 'test_db')
    
    def test_get_methods(self):
        """Test config getter methods"""
        # Test various getters
        self.assertTrue(self.config_manager.is_sandbox())
        self.assertEqual(self.config_manager.get_symbols(), ['BTC/USD', 'ETH/USD'])
        self.assertEqual(self.config_manager.get_timeframes(), ['1h', '4h'])
    
    def test_get_by_path(self):
        """Test getting config by path"""
        # Test get method
        self.assertEqual(self.config_manager.get('database.name'), 'test_db')
        self.assertEqual(self.config_manager.get('exchange.api_key'), 'test_key')
        self.assertEqual(self.config_manager.get('nonexistent.key', 'default'), 'default')