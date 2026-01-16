# src/components/postgres_writer.py
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import os
import sys
from typing import Dict, Any
import logging

from src.exception import CustomException
from src.logger import logging

class PostgreSQLWriter:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize PostgreSQL writer for customer segmentation data."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # PostgreSQL configuration
            pg_config = self.config.get('postgresql', {})
            self.host = pg_config.get('host', os.getenv('PG_HOST'))
            self.database = pg_config.get('database', os.getenv('PG_DATABASE'))
            self.user = pg_config.get('user', os.getenv('PG_USER'))
            self.password = pg_config.get('password', os.getenv('PG_PASSWORD'))
            self.port = pg_config.get('port', os.getenv('PG_PORT'))
            self.schema = pg_config.get('schema', os.getenv('PG_SCHEMA', 'data_science'))
            
            self.connection = None
            logging.info("PostgreSQLWriter initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize PostgreSQLWriter: {e}")
            raise CustomException(e, sys)

    def connect(self):
        """Establish connection to PostgreSQL."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.connection.autocommit = False
            logging.info(f"Connected to PostgreSQL: {self.host}:{self.port}/{self.database}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logging.info("PostgreSQL connection closed")
    
    def get_recommendation(self, segment: str) -> str:
        """Get recommendation based on segment using exact mapping."""
        recommendations = {
            "Low Risk - High Value": "Prioritize & reward",
            "Low Risk - Low Value": "Grow value",
            "Medium Risk - High Value": "Monitor risk, retain",
            "Medium Risk - Low Value": "Control exposure",
            "High Risk": "Restrict / red-flag"
        }
        return recommendations.get(segment, "Further analysis required")
    
    def decode_demographic(self, field_name: str, code: str) -> str:
        """Decode demographic codes to human-readable values."""
        
        # Marital status mapping
        marital_status_map = {
            '0': 'divorced',
            '1': 'married',
            '2': 'others',
            '3': 'separated',
            '4': 'single',
            '5': 'widowed'
        }
        
        # Onboarding channel mapping
        channel_key_map = {
            '0': 'android',
            '1': 'FSA',
            '2': 'IOS',
            '3': 'oxygen_now',
            '4': 'TSA',
            '5': 'web',
            '6': 'oxygen'
        }
        
        # Gender mapping (assuming 0=female, 1=male based on your description)
        gender_map = {
            '0': 'female',
            '1': 'male'
        }
        
        # Employment status mapping
        employment_map = {
            '0': 'employed',
            '1': 'others',
            '2': 'self_employed',
            '3': 'unemployed'
        }
        
        # Purpose mapping
        purpose_map = {
            '0': 'auto/transportation',
            '1': 'consumer_goods',
            '2': 'education',
            '3': 'electronics',
            '4': 'emergency_funds',
            '5': 'events',
            '6': 'healthcare',
            '7': 'housing',
            '8': 'personal',
            '9': 'retail/home_goods',
            '10': 'transportation',
            '11': 'travel_&_hospitality',
            '12': 'uncategorized',
            '13': 'utilities'
        }
        
        # Income bracket - keep as is but remove "income_bracket:" prefix
        if field_name == 'income_bracket':
            return f"income: {code}"
        
        # Age category - change prefix to "age:"
        if field_name == 'age_category':
            return f"age: {code}"
        
        # Apply mappings
        if field_name == 'marital_status':
            return marital_status_map.get(str(code).strip(), f"marital_status: {code}")
        elif field_name == 'dw_channel_key':
            return channel_key_map.get(str(code).strip(), f"channel: {code}")
        elif field_name == 'gender':
            return gender_map.get(str(code).strip(), f"gender: {code}")
        elif field_name == 'employment_status':
            return employment_map.get(str(code).strip(), f"employment: {code}")
        elif field_name == 'purpose':
            return purpose_map.get(str(code).strip(), f"for_{purpose_map.get(str(code).strip(), code)}")
        
        # For other fields, just return as is
        return f"{field_name}: {code}"
    
    def parse_combination(self, combination_str: str) -> str:
        """Parse combination string and convert to human-readable format with commas."""
        if not combination_str or pd.isna(combination_str):
            return ""
        
        parts = combination_str.split(' | ')
        decoded_parts = []
        
        for part in parts:
            if ': ' in part:
                field, value = part.split(': ', 1)
                field = field.strip()
                value = value.strip()
                
                # Decode the demographic
                decoded = self.decode_demographic(field, value)
                decoded_parts.append(decoded)
            else:
                decoded_parts.append(part)
        
        # Join with commas instead of |
        return ', '.join(decoded_parts)
    
    def write_customer_segments(self, segmented_df: pd.DataFrame):
        """
        Write customer segmentation results to customer_segment table.
        
        Args:
            segmented_df: DataFrame from segmenter.py with columns: customer_id, segment
        """
        if not self.connection:
            if not self.connect():
                raise ConnectionError("Could not connect to PostgreSQL")
        
        cursor = self.connection.cursor()
        
        try:
            logging.info(f"Writing customer segments to {self.schema}.customer_segment...")
            
            # Prepare data for insertion
            records = []
            for _, row in segmented_df.iterrows():
                # Convert customer_id to string
                customer_id = str(row['customer_id'])
                
                # Handle segment value
                segment = row['segment']
                if hasattr(segment, 'value'):  # If it's an enum
                    segment_str = segment.value
                else:
                    segment_str = str(segment)
                
                # Get recommendation using exact mapping
                recommendation = self.get_recommendation(segment_str)
                
                records.append((
                    customer_id,      # customer_id as VARCHAR
                    segment_str,      # segment_assigned
                    recommendation    # recommendation
                ))
            
            if not records:
                logging.warning("No records to write")
                return
            
            # SQL query for customer_segment table (new structure)
            query = f"""
            INSERT INTO {self.schema}.customer_segment 
            (customer_id, segment_assigned, recommendation)
            VALUES (%s, %s, %s)
            ON CONFLICT (customer_id) DO UPDATE SET
                segment_assigned = EXCLUDED.segment_assigned,
                recommendation = EXCLUDED.recommendation,
                updated_at = CURRENT_TIMESTAMP;
            """
            
            # Execute in batches
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                execute_batch(cursor, query, batch)
                logging.info(f"  Processed batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}")
            
            self.connection.commit()
            logging.info(f"✅ Successfully wrote {len(records)} customer segments to {self.schema}.customer_segment")
            
            # Log segment distribution
            cursor.execute(f"""
                SELECT segment_assigned, COUNT(*) as customer_count
                FROM {self.schema}.customer_segment
                GROUP BY segment_assigned
                ORDER BY customer_count DESC;
            """)
            
            segments = cursor.fetchall()
            for segment, count in segments:
                logging.info(f"  {segment}: {count} customers")
            
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Failed to write customer segments: {str(e)}")
            raise CustomException(e, sys)
        finally:
            if cursor:
                cursor.close()
    
    def write_segment_profiles(self, profiles_df: pd.DataFrame):
        """
        Write segment profiles to segment_profile table with formatted output.
        
        Args:
            profiles_df: DataFrame from profile_analyzer.py with segment profiles
        """
        if not self.connection:
            if not self.connect():
                raise ConnectionError("Could not connect to PostgreSQL")
        
        cursor = self.connection.cursor()
        
        try:
            logging.info(f"Writing segment profiles to {self.schema}.segment_profile...")
            
            # Clear existing data
            clear_query = f"DELETE FROM {self.schema}.segment_profile;"
            cursor.execute(clear_query)
            logging.info("  Cleared existing segment profiles")
            
            # Group profiles by segment and format them
            segment_profiles = {
                'low_risk_high_value': '',
                'low_risk_low_value': '',
                'medium_risk_high_value': '',
                'medium_risk_low_value': '',
                'high_risk': ''
            }
            
            # Process each profile
            for _, row in profiles_df.iterrows():
                segment = row['segment']
                if hasattr(segment, 'value'):
                    segment_name = segment.value
                else:
                    segment_name = str(segment)
                
                # Parse and format the combination
                combination_str = row.get('combination', '')
                formatted_combination = self.parse_combination(combination_str)
                
                # Only include the formatted combination (no profile ID, no stats)
                if formatted_combination:
                    # Map segment to column
                    if segment_name == "Low Risk - High Value":
                        if segment_profiles['low_risk_high_value']:
                            segment_profiles['low_risk_high_value'] += ", "
                        segment_profiles['low_risk_high_value'] += formatted_combination
                    elif segment_name == "Low Risk - Low Value":
                        if segment_profiles['low_risk_low_value']:
                            segment_profiles['low_risk_low_value'] += ", "
                        segment_profiles['low_risk_low_value'] += formatted_combination
                    elif segment_name == "Medium Risk - High Value":
                        if segment_profiles['medium_risk_high_value']:
                            segment_profiles['medium_risk_high_value'] += ", "
                        segment_profiles['medium_risk_high_value'] += formatted_combination
                    elif segment_name == "Medium Risk - Low Value":
                        if segment_profiles['medium_risk_low_value']:
                            segment_profiles['medium_risk_low_value'] += ", "
                        segment_profiles['medium_risk_low_value'] += formatted_combination
                    elif segment_name == "High Risk":
                        if segment_profiles['high_risk']:
                            segment_profiles['high_risk'] += ", "
                        segment_profiles['high_risk'] += formatted_combination
            
            # Insert the profile data
            query = f"""
            INSERT INTO {self.schema}.segment_profile 
            (low_risk_high_value, low_risk_low_value, medium_risk_high_value, 
             medium_risk_low_value, high_risk)
            VALUES (%s, %s, %s, %s, %s);
            """
            
            cursor.execute(query, (
                segment_profiles['low_risk_high_value'].strip() or '',
                segment_profiles['low_risk_low_value'].strip() or '',
                segment_profiles['medium_risk_high_value'].strip() or '',
                segment_profiles['medium_risk_low_value'].strip() or '',
                segment_profiles['high_risk'].strip() or ''
            ))
            
            self.connection.commit()
            logging.info(f"✅ Successfully wrote formatted segment profiles to {self.schema}.segment_profile")
            
            # Show what was written
            cursor.execute(f"""
                SELECT 
                    low_risk_high_value,
                    low_risk_low_value,
                    medium_risk_high_value,
                    medium_risk_low_value,
                    high_risk
                FROM {self.schema}.segment_profile;
            """)
            
            result = cursor.fetchone()
            if result:
                logging.info("\n📋 Formatted profiles written:")
                for i, (col_name, value) in enumerate(zip(
                    ['Low Risk High Value', 'Low Risk Low Value', 'Medium Risk High Value', 
                     'Medium Risk Low Value', 'High Risk'], result
                )):
                    if value:
                        logging.info(f"  {col_name}: {value}")
            
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Failed to write segment profiles: {str(e)}")
            raise CustomException(e, sys)
        finally:
            if cursor:
                cursor.close()