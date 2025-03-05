#!/usr/bin/env python
"""
Script to gracefully shut down the trading system.
"""

import os
import signal
import sys
import logging
from pathlib import Path

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def shutdown_system():
    """Gracefully shut down the trading system"""
    try:
        # Check if process IDs file exists
        pid_file = Path('process_ids.txt')
        if not pid_file.exists():
            logger.error("Process IDs file not found. Is the system running?")
            return False
        
        # Read process IDs
        pids = {}
        with open(pid_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=')
                    pids[key] = int(value)
        
        # Send termination signal to each process
        for name, pid in pids.items():
            try:
                logger.info(f"Terminating {name} (PID: {pid})...")
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to {name}")
            except ProcessLookupError:
                logger.warning(f"Process {name} (PID: {pid}) not found.")
            except Exception as e:
                logger.error(f"Error terminating {name}: {str(e)}")
        
        # Remove the PID file
        pid_file.unlink(missing_ok=True)
        
        logger.info("Trading system shut down successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error shutting down trading system: {str(e)}")
        return False

if __name__ == "__main__":
    shutdown_system()