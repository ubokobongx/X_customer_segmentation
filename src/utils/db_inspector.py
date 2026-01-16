# src/utils/db_inspector.py
import psycopg2
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class PostgreSQLInspector:
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 5432):
        self.connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
    
    def get_tables_in_schema(self, schema: str = "data_science") -> List[str]:
        """Get all tables in a specific schema."""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = %s
        ORDER BY table_name;
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (schema,))
            tables = [row[0] for row in cursor.fetchall()]
        
        return tables
    
    def get_table_structure(self, schema: str, table_name: str) -> pd.DataFrame:
        """Get column details for a specific table."""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (schema, table_name))
            columns = cursor.description
            rows = cursor.fetchall()
        
        df = pd.DataFrame(rows, columns=[col[0] for col in columns])
        return df
    
    def get_sample_data(self, schema: str, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a table."""
        query = f'SELECT * FROM "{schema}"."{table_name}" LIMIT {limit};'
        
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    def close(self):
        self.connection.close()

# Usage script
def inspect_postgresql():
    inspector = PostgreSQLInspector(
        host="ox.c5eus2qa6gyy.eu-west-1.rds.amazonaws.com",
        database="postgres",  # or your database name
        user="ubokobong",
        password="4dcwG7H9ay1G",
        port=5432
    )
    
    print("=" * 60)
    print("CHECKING TABLES IN data_science SCHEMA")
    print("=" * 60)
    
    # 1. Get all tables in data_science schema
    tables = inspector.get_tables_in_schema("data_science")
    print(f"Tables in data_science schema: {tables}")
    
    # 2. Check structure of each table
    for table in tables:
        print(f"\n{'='*40}")
        print(f"Table: {table}")
        print(f"{'='*40}")
        
        structure = inspector.get_table_structure("data_science", table)
        print("Structure:")
        print(structure.to_string())
        
        # 3. Show sample data if table has data
        try:
            sample = inspector.get_sample_data("data_science", table, limit=3)
            print(f"\nSample data (first 3 rows):")
            print(sample.to_string())
        except Exception as e:
            print(f"\nNo data or error reading table: {e}")
    
    inspector.close()

if __name__ == "__main__":
    inspect_postgresql()