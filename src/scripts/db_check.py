#!/usr/bin/env python
"""
Script to check database connectivity and initialize database if needed.
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import the modules
from ..core.logger import log_manager
from ..core.db_init import DatabaseInitializer
from ..data.database import db

logger = log_manager.get_logger(__name__)

def check_database():
    """Check database connectivity and initialize if needed"""
    try:
        logger.info("Checking database connectivity...")
        
        # Initialize database
        initializer = DatabaseInitializer()
        if initializer.initialize_database():
            logger.info("Database initialized successfully")
        else:
            logger.error("Failed to initialize database")
            return False
        
        # Check database health
        health = db.check_database_health()
        logger.info(f"Database health: {health['status']}")
        logger.info(f"Response time: {health['response_time_ms']}ms")
        logger.info(f"Table counts: {health['table_counts']}")
        
        if health['status'] != 'healthy':
            logger.error(f"Database health check failed: {health['last_error']}")
            return False
        
        # Clear any test data if requested
        if len(sys.argv) > 1 and sys.argv[1] == '--clear-test-data':
            logger.info("Clearing test data...")
            with db.get_session() as session:
                session.execute("DELETE FROM trades WHERE strategy = 'test_strategy'")
                logger.info("Test data cleared")
        
        return True
        
    except Exception as e:
        logger.error(f"Database check failed: {str(e)}")
        return False

if __name__ == "__main__":
    if check_database():
        logger.info("Database check successful")
        sys.exit(0)
    else:
        logger.error("Database check failed")
        sys.exit(1)