"""
Centralized logging system for the trading platform.
Implements structured logging with different handlers and formatters.
"""

import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger
import sys

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

class LogManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.log_dir = Path('logs')
            self.log_dir.mkdir(exist_ok=True)
            self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration with multiple handlers"""
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # JSON file handler
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'trading.json',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        json_handler.setLevel(logging.INFO)
        json_formatter = CustomJsonFormatter()
        json_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'error.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s\n'
            'Exception:\n%(exc_info)s'
        )
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
        
        # Trade logger
        trade_logger = logging.getLogger('trades')
        trade_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'trades.json',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        trade_handler.setFormatter(CustomJsonFormatter())
        trade_logger.addHandler(trade_handler)
        
        # Performance logger
        perf_logger = logging.getLogger('performance')
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'performance.json',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        perf_handler.setFormatter(CustomJsonFormatter())
        perf_logger.addHandler(perf_handler)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance with the specified name"""
        return logging.getLogger(name)
    
    def log_trade(self, trade_data: dict):
        """Log trade information"""
        logger = logging.getLogger('trades')
        logger.info('', extra=trade_data)
    
    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        logger = logging.getLogger('performance')
        logger.info('', extra={
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        })
    
    def log_error(self, error: Exception, context: dict = None):
        """Log an error with additional context"""
        logger = logging.getLogger('errors')
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        logger.error(json.dumps(error_data), exc_info=True)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for log_file in self.log_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logging.info(f"Deleted old log file: {log_file}")
        except Exception as e:
            logging.error(f"Error cleaning up logs: {str(e)}")

# Global logging instance
log_manager = LogManager()