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
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            user=config['database']['user'],
            password=config['database']['password']
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
        exchange = ccxt.kraken({
            'apiKey': config['exchange']['api_key'],
            'secret': config['exchange']['api_secret'],
            'enableRateLimit': True
        })
        
        if config['environment'] == 'sandbox':
            exchange.set_sandbox_mode(True)
        
        # Test API connection
        exchange.fetch_balance()
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deploy trading system')
    parser.add_argument(
        '--mode',
        choices=['full', 'backtest', 'paper'],
        default='paper',
        help='Deployment mode'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    deploy_system(args)