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
import time


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
    directories = ['data', 'logs', 'results', 'config', 'sessions']
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
            start_full_system(config)
        elif args.mode == 'backtest':
            start_backtest_mode()
        elif args.mode == 'paper':
            # Always use real data for paper trading
            logger.info("Starting paper trading with real market data")
            start_paper_trading(config, use_real_data=True)
        
        logger.info("System deployment completed successfully")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


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

def start_paper_trading(config, use_real_data=True):
    """Start system in paper trading mode with real market data"""
    try:
        # Set environment to sandbox
        os.environ['TRADING_ENV'] = 'sandbox'
        
        # Start paper trading components
        cmd = [
            sys.executable,
            '-m',
            'src.scripts.run',
            '--paper'
        ]
        
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Started system in paper trading mode with real market data")
        
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

def start_full_system(config):
    """Start all system components with improved process management"""
    try:
        # Start data streaming
        stream_process = subprocess.Popen(
            [sys.executable, '-m', 'src.data.stream'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Started data streaming (PID: {stream_process.pid})")
        
        # Wait a bit for the stream to initialize
        time.sleep(2)
        
        # Start trading system with the appropriate mode
        if os.environ.get('TRADING_ENV') == 'sandbox':
            trading_process = subprocess.Popen(
                [sys.executable, '-m', 'src.scripts.run', '--paper'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        else:
            trading_process = subprocess.Popen(
                [sys.executable, '-m', 'src.scripts.run'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        logger.info(f"Started trading system (PID: {trading_process.pid})")
        
        # Wait a bit for the trading system to initialize
        time.sleep(2)
        
        # Start monitoring dashboard with host binding
        dashboard_process = subprocess.Popen(
            [sys.executable, '-m', 'src.monitoring.dashboard', '--host', '0.0.0.0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Started monitoring dashboard (PID: {dashboard_process.pid})")
        
        # Save PIDs to file for later monitoring
        with open('process_ids.txt', 'w') as f:
            f.write(f"stream_pid={stream_process.pid}\n")
            f.write(f"trading_pid={trading_process.pid}\n")
            f.write(f"dashboard_pid={dashboard_process.pid}\n")
        
        # Check if processes are still running after a short delay
        time.sleep(5)
        
        processes = [
            ('Data Stream', stream_process),
            ('Trading System', trading_process),
            ('Dashboard', dashboard_process)
        ]
        
        all_running = True
        for name, process in processes:
            if process.poll() is not None:
                # Process has terminated
                stdout, stderr = process.communicate()
                logger.error(f"{name} process terminated unexpectedly")
                logger.error(f"STDOUT: {stdout.decode('utf-8')}")
                logger.error(f"STDERR: {stderr.decode('utf-8')}")
                all_running = False
        
        if all_running:
            logger.info("All components started successfully!")
            
            # Print dashboard URL
            dashboard_port = config.get('monitoring', {}).get('dashboard_port', 8050)
            logger.info(f"Dashboard is available at: http://localhost:{dashboard_port}")
        else:
            logger.error("Some components failed to start. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Error starting system components: {e}")
        raise

# Add this at the end of deploy.py
if __name__ == "__main__":
    args = parse_arguments()
    deploy_system(args)