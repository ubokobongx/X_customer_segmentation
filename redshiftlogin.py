from sqlalchemy.engine import Engine
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2

class RedshiftConnector:
    def __init__(self):
        self.config = {
            'dbname': 'ox_datawarehouse',
            'user': 'oxygen_qa',
            'password': 'OXygenDataus3r',
            'host': 'ox-datawarehouse-cls.chkhpbcphj8j.eu-west-1.redshift.amazonaws.com',
            'port': '5439'
        }
        # Disable the problematic PostgreSQL parameter check
        PGDialect_psycopg2._set_backslash_escapes = lambda self, connection: None

    def get_engine(self) -> Engine:
        connection_string = (
            f"postgresql+psycopg2://{self.config['user']}:{self.config['password']}@"
            f"{self.config['host']}:{self.config['port']}/{self.config['dbname']}"
        )
        return create_engine(
            connection_string,
            connect_args={
                'sslmode': 'prefer',
                'connect_timeout': 15,  # 15 second timeout
                'keepalives': 1,
                'keepalives_idle': 30,
                'keepalives_interval': 10,
                'keepalives_count': 5
            },
            pool_pre_ping=True,  # Test connections before use
            echo=False  # Set to True for debugging queries
        )