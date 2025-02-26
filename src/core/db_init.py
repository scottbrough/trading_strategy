"""
Database initialization module for creating and setting up the database.
Handles first-time setup and schema validation.
"""

import os
import subprocess
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, inspect
import alembic.config

from ..core.config import config
from ..core.logger import log_manager

logger = log_manager.get_logger(__name__)

class DatabaseInitializer:
    def __init__(self):
        self.db_config = config.db_config
        self.connection_string = f"postgresql://{self.db_config.user}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}"
        
    def initialize_database(self):
        """Initialize database and run migrations"""
        try:
            # Check if database exists
            if not self._database_exists():
                logger.info(f"Creating database {self.db_config.name}")
                self._create_database()
            
            # Connect to database
            engine = create_engine(f"{self.connection_string}/{self.db_config.name}")
            
            # Check schema and run migrations
            if not self._check_schema(engine):
                logger.info("Running database migrations")
                self._run_migrations()
                
            logger.info("Database initialization complete")
            return True
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            return False
            
    def _database_exists(self):
        """Check if database exists"""
        try:
            # Connect to default postgres database
            conn = psycopg2.connect(
                f"{self.connection_string}/postgres",
                connect_timeout=5
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Check if database exists
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", 
                              (self.db_config.name,))
                exists = cursor.fetchone() is not None
            
            conn.close()
            return exists
        except Exception as e:
            logger.error(f"Error checking database existence: {str(e)}")
            raise
            
    def _create_database(self):
        """Create database if it doesn't exist"""
        try:
            # Connect to default postgres database
            conn = psycopg2.connect(
                f"{self.connection_string}/postgres",
                connect_timeout=5
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Create database
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE {self.db_config.name}")
            
            conn.close()
            logger.info(f"Created database {self.db_config.name}")
            return True
        except Exception as e:
            logger.error(f"Error creating database: {str(e)}")
            raise
            
    def _check_schema(self, engine):
        """Check if database schema exists"""
        try:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            required_tables = ['trades', 'ohlcv', 'performance']
            
            return all(table in tables for table in required_tables)
        except Exception as e:
            logger.error(f"Error checking schema: {str(e)}")
            return False
            
    def _run_migrations(self):
        """Run Alembic migrations"""
        try:
            # Get alembic config
            alembic_cfg = alembic.config.Config("alembic.ini")
            
            # Run migrations
            subprocess.check_call(['alembic', 'upgrade', 'head'])
            
            logger.info("Database migrations complete")
            return True
        except Exception as e:
            logger.error(f"Error running migrations: {str(e)}")
            raise