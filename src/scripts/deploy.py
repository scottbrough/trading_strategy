"""
Deployment script for the trading system.
Handles setup, database initialization, and system deployment.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
import yaml
import psycopg2
from datetime import datetime
import logging


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/deployment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def check_dependencies():
    """Check if all required dependencies are installed"""
    required = [
        'python-dotenv',
        'pandas',
        'numpy',
        'sqlalchemy',
        'psycopg2-binary',
        'ta-lib',
        'dash',
        'plotly',
        'websockets',
        'ccxt'
    ]
    
    logger.info("Checking dependencies...")
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Installing missing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

def setup_database(config):
    """Initialize database and run migrations"""
    try:
        # Connect to PostgreSQL's default database instead
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            user=config['database']['user'],
            password=config['database']['password'],
            database="postgres"  # Connect to default database instead
        )
        conn.autocommit = True
        
        # Create database if not exists
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database']['name'],))
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {config['database']['name']}")
                logger.info(f"Created database {config['database']['name']}")
        
        # Run migrations
        logger.info("Running database migrations...")
        subprocess.check_call(['alembic', 'upgrade', 'head'])
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

def check_api_access(config):
    """Verify API access and credentials"""
    try:
        import ccxt
        
        # For sandbox/demo mode, we'll bypass the direct API check
        if config['environment'] == 'sandbox':
            logger.info("Running in sandbox mode - skipping direct API verification")
            
            # Check if API credentials are provided
            if not config['exchange']['api_key'] or not config['exchange']['api_secret']:
                logger.warning("API credentials are missing or empty")
            
            # Just verify we can import the required libraries
            import websockets
            import json
            import hmac
            import hashlib
            
            logger.info("Required libraries for API access are available")
            return
        
        # Only perform direct API checks in production mode
        exchange = getattr(ccxt, config['exchange']['name'].lower())({
            'apiKey': config['exchange']['api_key'],
            'secret': config['exchange']['api_secret'],
            'enableRateLimit': True
        })
        
        # Test API connection
        exchange.fetch_ticker('BTC/USD')
        logger.info("API access verified successfully")
        
    except Exception as e:
        logger.error(f"API access verification failed: {e}")
        raise

def setup_directories():
    """Create required directories"""
    directories = ['data', 'logs', 'results', 'config']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_name}")

def deploy_system(args):
    """Deploy the trading system"""
    try:
        logger.info("Starting system deployment...")
        
        # Load configuration
        with open('config/config.yaml') as f:
            config = yaml.safe_load(f)
        
        # Setup steps
        setup_directories()
        check_dependencies()
        setup_database(config)
        check_api_access(config)
        
        # Start system components
        if args.mode == 'full':
            start_full_system()
        elif args.mode == 'backtest':
            start_backtest_mode()
        elif args.mode == 'paper':
            start_paper_trading()
        
        logger.info("System deployment completed successfully")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

def start_full_system():
    """Start all system components"""
    try:
        # Start data streaming
        subprocess.Popen([sys.executable, '-m', 'src.data.stream'])
        logger.info("Started data streaming")
        
        # Start trading system
        subprocess.Popen([sys.executable, '-m', 'src.scripts.run'])
        logger.info("Started trading system")
        
        # Start monitoring dashboard
        subprocess.Popen([sys.executable, '-m', 'src.monitoring.dashboard'])
        logger.info("Started monitoring dashboard")
        
    except Exception as e:
        logger.error(f"Error starting system components: {e}")
        raise

def start_backtest_mode():
    """Start system in backtest mode"""
    try:
        subprocess.Popen([
            sys.executable,
            '-m',
            'src.scripts.backtest',
            '--config',
            'config/config.yaml'
        ])
        logger.info("Started system in backtest mode")
        
    except Exception as e:
        logger.error(f"Error starting backtest mode: {e}")
        raise

def start_paper_trading():
    """Start system in paper trading mode"""
    try:
        # Set environment to sandbox
        os.environ['TRADING_ENV'] = 'sandbox'
        
        # Start paper trading components
        start_full_system()
        logger.info("Started system in paper trading mode")
        
    except Exception as e:
        logger.error(f"Error starting paper trading: {e}")
        raise

def parse_arguments():
    """Parse command line arguments with improved options"""
    parser = argparse.ArgumentParser(description='Deploy trading system')
    
    # Main operation mode
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest'],
        default='paper',
        help='Deployment mode (paper trading, live trading, or backtest)'
    )
    
    # Strategy configuration
    parser.add_argument(
        '--strategy',
        type=str,
        help='Strategy to run (e.g., momentum_strategy.MomentumStrategy)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to strategy configuration file'
    )
    
    # Data options
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Symbols to trade (e.g., BTC/USD ETH/USD)'
    )
    
    parser.add_argument(
        '--timeframes',
        type=str,
        nargs='+',
        default=['1h'],
        help='Timeframes to use (e.g., 1m 5m 1h)'
    )
    
    # Connection options
    parser.add_argument(
        '--exchange',
        type=str,
        default='kraken',
        help='Exchange to use'
    )
    
    # Monitoring options
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8050,
        help='Port for monitoring dashboard'
    )
    
    # Database options
    parser.add_argument(
        '--db-host',
        type=str,
        help='Database host'
    )
    
    parser.add_argument(
        '--db-port',
        type=int,
        help='Database port'
    )
    
    return parser.parse_args()