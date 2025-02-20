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
        
        # File handler for JSON logs
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
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance with the specified name"""
        return logging.getLogger(name)
    
    def log_trade(self, trade_data: dict):
        """Log trade information to a separate trade log file"""
        trade_logger = logging.getLogger('trades')
        if not trade_logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                self.log_dir / 'trades.json',
                maxBytes=10*1024*1024,
                backupCount=5
            )
            handler.setFormatter(CustomJsonFormatter())
            trade_logger.addHandler(handler)
        
        trade_logger.info('', extra=trade_data)
    
    def log_error(self, error: Exception, context: dict = None):
        """Log an error with additional context"""
        logger = logging.getLogger('errors')
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        logger.error(json.dumps(error_data), exc_info=True)

# Global logging instance
log_manager = LogManager()

# Example usage:
if __name__ == "__main__":
    logger = log_manager.get_logger(__name__)
    
    # Log different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    
    # Log an error
    try:
        raise ValueError("Example error")
    except Exception as e:
        log_manager.log_trade(trade_data)
    
    def setup_performance_logging(self):
        """Setup performance logging for metrics and analysis"""
        perf_logger = logging.getLogger('performance')
        if not perf_logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                self.log_dir / 'performance.json',
                maxBytes=10*1024*1024,
                backupCount=5
            )
            handler.setFormatter(CustomJsonFormatter())
            perf_logger.addHandler(handler)
    
    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        perf_logger = logging.getLogger('performance')
        perf_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        }
        perf_logger.info('', extra=perf_data)
    
    def log_system_health(self, health_data: dict):
        """Log system health metrics"""
        health_logger = logging.getLogger('system_health')
        if not health_logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                self.log_dir / 'system_health.json',
                maxBytes=10*1024*1024,
                backupCount=5
            )
            handler.setFormatter(CustomJsonFormatter())
            health_logger.addHandler(handler)
        
        health_data['timestamp'] = datetime.utcnow().isoformat()
        health_logger.info('', extra=health_data)
    
    def setup_audit_logging(self):
        """Setup audit logging for system changes and critical operations"""
        audit_logger = logging.getLogger('audit')
        if not audit_logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                self.log_dir / 'audit.json',
                maxBytes=10*1024*1024,
                backupCount=10
            )
            handler.setFormatter(CustomJsonFormatter())
            audit_logger.addHandler(handler)
    
    def log_audit(self, action: str, details: dict):
        """Log audit information for system changes"""
        audit_logger = logging.getLogger('audit')
        audit_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'details': details
        }
        audit_logger.info('', extra=audit_data)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    logging.info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    logging.error(f"Error deleting log file {log_file}: {str(e)}")
    
    def get_recent_errors(self, hours: int = 24) -> list:
        """Get recent error logs"""
        error_logs = []
        error_file = self.log_dir / 'error.log'
        
        if not error_file.exists():
            return error_logs
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with open(error_file, 'r') as f:
            for line in f:
                try:
                    log_time_str = line.split(' - ')[0]
                    log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S,%f')
                    if log_time >= cutoff_time:
                        error_logs.append(line.strip())
                except:
                    continue
        
        return error_logs

# Example of additional usage:
if __name__ == "__main__":
    logger = log_manager.get_logger(__name__)
    
    # Log performance metrics
    performance_metrics = {
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.15,
        'win_rate': 0.65,
        'profit_factor': 1.8
    }
    log_manager.log_performance(performance_metrics)
    
    # Log system health
    health_data = {
        'cpu_usage': 45.2,
        'memory_usage': 62.8,
        'disk_space': 78.5,
        'api_latency': 125
    }
    log_manager.log_system_health(health_data)
    
    # Log audit event
    audit_details = {
        'user': 'system',
        'config_change': 'risk_parameters',
        'old_value': {'max_position_size': 0.1},
        'new_value': {'max_position_size': 0.15}
    }
    log_manager.log_audit('config_update', audit_details)
log_error(e, {'context': 'example'})
    
    # Log a trade
    trade_data = {
        'symbol': 'BTC/USD',
        'side': 'buy',
        'price': 50000,
        'amount': 0.1,
        'timestamp': datetime.utcnow().isoformat()
    }
    log_manager.