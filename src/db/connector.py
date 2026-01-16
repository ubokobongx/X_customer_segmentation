# src/db/connector.py
import os
import sys
from typing import Dict, Optional
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
import yaml
from dotenv import load_dotenv
from src.exception import CustomException
from src.logger import logging

class RedshiftConnector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Secure Redshift connector using environment variables."""
        try:
            # Load environment variables from .env file
            load_dotenv()
            
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Get credentials from environment variables (SECURE)
            self.db_config = {
                'dbname': self.config['database']['dbname'],
                'user': os.getenv('REDSHIFT_USER', 'oxygen_qa'),
                'password': os.getenv('REDSHIFT_PASSWORD'),  # No default - fail if missing
                'host': self.config['database']['host'],
                'port': self.config['database']['port']
            }
            
            # Validate credentials
            if not self.db_config['password']:
                error_msg = "Redshift password not found in environment variables. Check your .env file."
                logging.error(error_msg)
                # Debug: Check what environment variables are available
                env_vars = {k: v for k, v in os.environ.items() if 'REDSHIFT' in k}
                logging.info(f"Available REDSHIFT environment variables: {env_vars}")
                raise ValueError(error_msg)
            
            # Log connection info (without showing password)
            logging.info(f"✓ Loaded Redshift credentials for user: {self.db_config['user']}")
            logging.info(f"✓ Connecting to: {self.db_config['host']}:{self.db_config['port']}")
            logging.info(f"✓ Database: {self.db_config['dbname']}")
            
            # Disable problematic PostgreSQL parameter check
            PGDialect_psycopg2._set_backslash_escapes = lambda self, connection: None
            
            logging.info("RedshiftConnector initialized successfully")
            
        except FileNotFoundError as e:
            logging.error(f"Configuration file not found: {config_path}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"Failed to initialize RedshiftConnector: {e}")
            raise CustomException(e, sys)

    def get_engine(self) -> Engine:
        """Create SQLAlchemy engine with secure connection."""
        try:
            connection_string = (
                f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
            )
            
            # Log connection string (with password masked for security)
            masked_connection = (
                f"postgresql+psycopg2://{self.db_config['user']}:{'*' * len(self.db_config['password'])}@"
                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
            )
            logging.info(f"Creating connection: {masked_connection}")
            
            engine = create_engine(
                connection_string,
                connect_args={
                    'sslmode': 'require',  # Always use SSL for production
                    'connect_timeout': 30,
                    'keepalives': 1,
                    'keepalives_idle': 30,
                    'keepalives_interval': 10,
                    'keepalives_count': 5
                },
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                echo=False
            )
            
            logging.info("✓ Redshift engine created successfully")
            return engine
            
        except Exception as e:
            logging.error(f"Failed to create Redshift engine: {e}")
            raise CustomException(e, sys)