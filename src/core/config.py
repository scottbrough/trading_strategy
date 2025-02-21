"""
Configuration management system for the trading platform.
Handles loading and validation of configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import logging
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    type: str
    host: str
    port: int
    name: str
    user: str
    password: str

@dataclass
class ExchangeConfig:
    name: str
    sandbox: bool
    api_url: str
    websocket_url: str
    rate_limit: float

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = logging.getLogger(__name__)
            self.config_dir = Path(__file__).parent.parent.parent / 'config'
            self.load_env()
            self.load_config()
    
    def load_env(self):
        """Load environment variables from .env file"""
        env_path = self.config_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            self.logger.warning(".env file not found, using system environment variables")
    
    def load_config(self):
        """Load and validate configuration from YAML files"""
        try:
            # Load main configuration
            config_path = self.config_dir / 'config.yaml'
            trading_config_path = self.config_dir / 'trading_config.yaml'
            
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
            
            with open(trading_config_path) as f:
                trading_config = yaml.safe_load(f)
                self.config.update(trading_config)
            
            # Substitute environment variables
            self._substitute_env_vars(self.config)
            
            # Validate configuration
            self._validate_config()
            
            # Create specific config objects
            self.db_config = DatabaseConfig(**self.config['database'])
            self.exchange_config = ExchangeConfig(**self.config['exchange'])
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _substitute_env_vars(self, config: Dict[str, Any]):
        """Recursively substitute environment variables in configuration"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var)
                if config[key] is None:
                    raise ValueError(f"Environment variable {env_var} not set")
    
    def _validate_config(self):
        """Validate configuration values"""
        required_sections = ['database', 'exchange', 'data', 'strategy', 'monitoring']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate database configuration
        db_config = self.config['database']
        if not all(k in db_config for k in ['type', 'host', 'port', 'name', 'user', 'password']):
            raise ValueError("Invalid database configuration")
        
        # Validate exchange configuration
        ex_config = self.config['exchange']
        if not all(k in ex_config for k in ['name', 'sandbox', 'api_url', 'websocket_url']):
            raise ValueError("Invalid exchange configuration")
        
        # Validate trading parameters
        if 'trading_params' not in self.config:
            raise ValueError("Missing trading parameters")
        
        # Validate optimization parameters
        if 'optimization_params' not in self.config:
            raise ValueError("Missing optimization parameters")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def is_sandbox(self) -> bool:
        """Check if running in sandbox mode"""
        return self.config['environment'] == 'sandbox'
    
    def get_symbols(self) -> list:
        """Get configured trading symbols"""
        return self.config['data']['symbols']
    
    def get_timeframes(self) -> list:
        """Get configured timeframes"""
        return self.config['data']['timeframes']
    
    def get_risk_params(self) -> dict:
        """Get risk management parameters"""
        return self.config['risk_params']
    
    def get_trading_params(self) -> dict:
        """Get trading parameters"""
        return self.config['trading_params']
    
    def get_optimization_params(self) -> dict:
        """Get optimization parameters"""
        return self.config['optimization_params']
    
    def get_monitoring_config(self) -> dict:
        """Get monitoring configuration"""
        return self.config['monitoring']

# Global configuration instance
config = ConfigManager()