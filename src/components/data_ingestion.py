# src/components/data_ingestion.py
"""
Data Ingestion Component for Customer Segmentation Pipeline.
Extracts data from Redshift using the predefined SQL query.
"""
import pandas as pd
import yaml
from sqlalchemy import text
from typing import Tuple, Dict, Any
import sys
import os
import json

from src.db.connector import RedshiftConnector
from src.db.queries import DATA_EXTRACTION_QUERY
from src.exception import CustomException
from src.logger import logging
from data.schemas import RawCustomerData


class DataIngestion:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data ingestion component.
        
        Args:
            config_path: Path to configuration YAML file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.raw_data_path = self.config['data']['raw_data_path']
            self.db_connector = RedshiftConnector(config_path)
            self.query = DATA_EXTRACTION_QUERY
            
            logging.info("DataIngestion initialized successfully")
            logging.info(f"Raw data path: {self.raw_data_path}")
            
        except FileNotFoundError as e:
            logging.error(f"Configuration file not found: {config_path}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"Failed to initialize DataIngestion: {e}")
            raise CustomException(e, sys)

    def extract_data(self) -> pd.DataFrame:
        """
        Extract data from Redshift using the predefined SQL query.
        
        Returns:
            DataFrame containing raw customer data
            
        Raises:
            CustomException: If data extraction fails
        """
        try:
            logging.info("Starting data extraction from Redshift...")
            logging.info(f"Using query length: {len(self.query)} characters")
            
            engine = self.db_connector.get_engine()
            
            # Log connection info (without credentials)
            logging.info(f"Connected to: {self.db_connector.db_config['host']}:{self.db_connector.db_config['port']}")
            logging.info(f"Database: {self.db_connector.db_config['dbname']}")
            
            with engine.begin() as conn:
                # Try SQLAlchemy execution with text()
                try:
                    df = pd.read_sql_query(text(self.query), conn)
                except TypeError as e:
                    # Fallback for older pandas/SQLAlchemy versions
                    logging.warning(f"Using fallback method: {e}")
                    df = pd.read_sql_query(self.query, conn.connection)
                except Exception as e:
                    logging.error(f"SQL execution failed: {e}")
                    raise
            
            # Validate that we got data
            if df.empty:
                logging.warning("No data returned from query")
            else:
                logging.info(f"✅ Successfully extracted {len(df)} records from Redshift")
                logging.info(f"Data shape: {df.shape}")
                logging.info(f"Columns extracted: {list(df.columns)}")
                
                # Show basic statistics
                if 'customer_id' in df.columns:
                    logging.info(f"Customer ID range: {df['customer_id'].min()} - {df['customer_id'].max()}")
                
                # Show sample of data types
                logging.info("Sample data types:")
                for col in list(df.columns)[:10]:  # First 10 columns
                    logging.info(f"  {col}: {df[col].dtype}")
            
            return df
            
        except Exception as e:
            logging.error(f"Data extraction failed: {e}")
            
            # Provide more detailed error information
            error_msg = f"""
            Data Extraction Error Details:
            - Error Type: {type(e).__name__}
            - Error Message: {str(e)}
            - Query Length: {len(self.query)} characters
            - Config Used: {self.db_connector.db_config['host']}:{self.db_connector.db_config['port']}/{self.db_connector.db_config['dbname']}
            """
            logging.error(error_msg)
            
            raise CustomException(e, sys)

    def pre_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-clean data before validation to handle common data issues.
        
        Args:
            df: Raw DataFrame from Redshift
            
        Returns:
            Pre-cleaned DataFrame
        """
        try:
            logging.info("Pre-cleaning data before validation...")
            df_clean = df.copy()
            
            # Handle tenor_in_months - round to nearest integer
            if 'tenor_in_months' in df_clean.columns:
                df_clean['tenor_in_months'] = pd.to_numeric(df_clean['tenor_in_months'], errors='coerce')
                df_clean['tenor_in_months'] = df_clean['tenor_in_months'].fillna(1)
                df_clean['tenor_in_months'] = df_clean['tenor_in_months'].round().astype(int)
                logging.info(f"Cleaned tenor_in_months: {df_clean['tenor_in_months'].min()}-{df_clean['tenor_in_months'].max()}")
            
            # Handle string fields - replace None with empty string
            string_fields = ['marital_status', 'state', 'location', 'purpose', 
                           'employment_status', 'dw_channel_key', 'gender']
            for field in string_fields:
                if field in df_clean.columns:
                    df_clean[field] = df_clean[field].fillna('')
                    # Convert to string type
                    df_clean[field] = df_clean[field].astype(str)
            
            # Clean gender field
            if 'gender' in df_clean.columns:
                df_clean['gender'] = df_clean['gender'].apply(
                    lambda x: 'Unknown' if str(x).strip().title() == 'Unknown' else str(x).strip().title()
                )
            
            # Ensure numeric fields are properly typed
            numeric_fields = ['total_missed_installment', 'total_amount_overdue', 
                            'total_loan_amount', 'ontime_repayment_rate', 'income',
                            'loan_count', 'maturity_dpd', 'count_14plus_dpd', 'age']
            
            for field in numeric_fields:
                if field in df_clean.columns:
                    df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
                    # Fill NaN with reasonable defaults
                    if field in ['total_missed_installment', 'total_amount_overdue', 
                                'maturity_dpd', 'count_14plus_dpd']:
                        df_clean[field] = df_clean[field].fillna(0)
                    elif field == 'ontime_repayment_rate':
                        df_clean[field] = df_clean[field].fillna(100).clip(0, 100)
                    elif field == 'income':
                        df_clean[field] = df_clean[field].fillna(df_clean[field].median())
                    elif field == 'age':
                        df_clean[field] = df_clean[field].fillna(30).clip(18, 100)
            
            # Ensure has_14plus_dpd is binary
            if 'has_14plus_dpd' in df_clean.columns:
                df_clean['has_14plus_dpd'] = pd.to_numeric(df_clean['has_14plus_dpd'], errors='coerce')
                df_clean['has_14plus_dpd'] = df_clean['has_14plus_dpd'].fillna(0).clip(0, 1).astype(int)
            
            logging.info(f"Pre-cleaning completed. Shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logging.error(f"Pre-cleaning failed: {e}")
            # Return original DataFrame if cleaning fails
            return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate extracted data against the RawCustomerData schema.
        
        Args:
            df: Raw DataFrame from Redshift
            
        Returns:
            DataFrame with validated records only
            
        Raises:
            CustomException: If validation fails catastrophically
        """
        try:
            logging.info("Starting data validation...")
            logging.info(f"Records to validate: {len(df)}")
            
            if df.empty:
                logging.warning("Empty DataFrame, nothing to validate")
                return df
            
            # Pre-clean the data before validation
            df_clean = self.pre_clean_data(df)
            
            validated_records = []
            validation_errors = []
            
            # Convert DataFrame to list of dicts for validation
            records = df_clean.to_dict('records')
            
            for i, record in enumerate(records):
                try:
                    # Validate against Pydantic schema
                    validated = RawCustomerData(**record)
                    validated_records.append(validated.dict())
                    
                    # Log progress every 1000 records
                    if (i + 1) % 1000 == 0:
                        logging.info(f"Validated {i + 1}/{len(records)} records")
                        
                except Exception as e:
                    validation_errors.append({
                        'record_index': i,
                        'customer_id': record.get('customer_id', 'Unknown'),
                        'error': str(e)
                    })
                    
                    # Log first few errors, then summarize
                    if len(validation_errors) <= 5:
                        logging.warning(f"Record {i} (Customer ID: {record.get('customer_id', 'Unknown')}) validation failed: {e}")
            
            # Create validated DataFrame
            validated_df = pd.DataFrame(validated_records)
            
            # Log validation results
            logging.info(f"✅ Validation complete:")
            logging.info(f"  - Valid records: {len(validated_df)}/{len(df)} ({len(validated_df)/len(df)*100:.1f}%)")
            logging.info(f"  - Invalid records: {len(validation_errors)}")
            
            if validation_errors:
                logging.warning(f"Validation errors summary:")
                for error in validation_errors[:10]:  # Show first 10 errors
                    logging.warning(f"  - Record {error['record_index']} (Customer {error['customer_id']}): {error['error']}")
                
                if len(validation_errors) > 10:
                    logging.warning(f"  ... and {len(validation_errors) - 10} more errors")
                
                # Save validation errors to file for analysis
                errors_df = pd.DataFrame(validation_errors)
                errors_path = os.path.join(os.path.dirname(self.raw_data_path), "validation_errors.csv")
                errors_df.to_csv(errors_path, index=False)
                logging.info(f"  - Validation errors saved to: {errors_path}")
            
            return validated_df
            
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            # Return original DataFrame if validation fails catastrophically
            logging.warning("Returning original DataFrame due to validation failure")
            return df

    def save_raw_data(self, df: pd.DataFrame, format: str = "csv") -> str:
        """
        Save raw data to disk in specified format.
        
        Args:
            df: DataFrame to save
            format: Output format ('parquet' or 'csv')
            
        Returns:
            Path to saved file
            
        Raises:
            CustomException: If save operation fails
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            
            if format.lower() == "parquet":
                # Use parquet for better performance and type preservation
                save_path = self.raw_data_path.replace('.csv', '.parquet')
                
                try:
                    # Try to save as parquet
                    df.to_parquet(save_path, index=False, compression='snappy')
                    logging.info(f"Raw data saved as Parquet to: {save_path}")
                    if os.path.exists(save_path):
                        logging.info(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
                except ImportError:
                    # If parquet library not available, fall back to CSV
                    logging.warning("Parquet library not available, falling back to CSV")
                    save_path = self.raw_data_path
                    df.to_csv(save_path, index=False)
                    logging.info(f"Raw data saved as CSV to: {save_path}")
                
            elif format.lower() == "csv":
                save_path = self.raw_data_path
                df.to_csv(save_path, index=False)
                logging.info(f"Raw data saved as CSV to: {save_path}")
                if os.path.exists(save_path):
                    logging.info(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
                
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'.")
            
            # Save metadata about the extraction
            metadata = {
                'extraction_timestamp': pd.Timestamp.now().isoformat(),
                'record_count': len(df),
                'columns': list(df.columns),
                'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'data_source': 'Redshift',
                'query_hash': hash(self.query) % 1000000,  # Simple hash for tracking
                'data_summary': self.generate_data_summary(df)
            }
            
            # Add customer_id range if available
            if 'customer_id' in df.columns:
                metadata['customer_id_range'] = {
                    'min': int(df['customer_id'].min()),
                    'max': int(df['customer_id'].max())
                }
            
            metadata_path = save_path.replace('.parquet', '_metadata.json').replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logging.info(f"Metadata saved to: {metadata_path}")
            
            return save_path
            
        except Exception as e:
            logging.error(f"Failed to save raw data: {e}")
            raise CustomException(e, sys)

    def generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the extracted data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            logging.info("Generating data summary...")
            
            summary = {
                'total_records': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'numeric_summary': {},
                'categorical_summary': {},
                'basic_statistics': {}
            }
            
            # Basic statistics for all columns
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    summary['basic_statistics'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std())
                    }
            
            # Numeric columns detailed summary
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col in df.columns:
                    summary['numeric_summary'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'zeros': int((df[col] == 0).sum()),
                        'missing': int(df[col].isna().sum())
                    }
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col in df.columns:
                    value_counts = df[col].value_counts()
                    summary['categorical_summary'][col] = {
                        'unique_values': int(value_counts.nunique()),
                        'top_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'top_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'missing': int(df[col].isna().sum())
                    }
            
            # Special fields analysis
            if 'has_14plus_dpd' in df.columns:
                default_rate = df['has_14plus_dpd'].mean()
                summary['default_analysis'] = {
                    'default_rate': float(default_rate),
                    'default_count': int(df['has_14plus_dpd'].sum()),
                    'non_default_count': int((df['has_14plus_dpd'] == 0).sum())
                }
            
            logging.info(f"Data summary generated: {len(df)} records, {len(df.columns)} columns")
            return summary
            
        except Exception as e:
            logging.warning(f"Could not generate complete data summary: {e}")
            return {'error': str(e)}

    def run(self) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        """
        Run the complete data ingestion pipeline.
        
        Returns:
            Tuple of (validated DataFrame, saved file path, summary statistics)
            
        Raises:
            CustomException: If any step of the pipeline fails
        """
        try:
            logging.info("=" * 80)
            logging.info("STARTING DATA INGESTION PIPELINE")
            logging.info("=" * 80)
            
            # Step 1: Extract data from Redshift
            logging.info("\n[1/3] EXTRACTING DATA FROM REDSHIFT")
            raw_df = self.extract_data()
            
            if raw_df.empty:
                logging.warning("No data extracted from Redshift. Pipeline may continue with empty dataset.")
                # Return empty but valid structure
                return raw_df, "", {'error': 'No data extracted'}
            
            # Step 2: Validate data
            logging.info("\n[2/3] VALIDATING DATA")
            validated_df = self.validate_data(raw_df)
            
            if validated_df.empty:
                logging.error("No valid records after validation. Pipeline cannot continue.")
                raise CustomException("No valid records after validation", sys)
            
            # Step 3: Save data - Use CSV for compatibility
            logging.info("\n[3/3] SAVING DATA")
            saved_path = self.save_raw_data(validated_df, format="csv")
            
            # Generate summary
            summary = self.generate_data_summary(validated_df)
            
            # Log completion
            logging.info("=" * 80)
            logging.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            logging.info(f"Summary:")
            logging.info(f"  - Valid records: {len(validated_df)}")
            logging.info(f"  - Saved to: {saved_path}")
            if 'default_analysis' in summary:
                logging.info(f"  - Default rate: {summary['default_analysis'].get('default_rate', 0):.1%}")
            
            return validated_df, saved_path, summary
            
        except Exception as e:
            logging.error("=" * 80)
            logging.error("DATA INGESTION PIPELINE FAILED")
            logging.error("=" * 80)
            logging.error(f"Error: {e}")
            raise CustomException(e, sys)


# Example usage and testing
if __name__ == "__main__":
    """Test the data ingestion component independently."""
    import sys
    import os
    
    # Add project root to path for testing
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        # Initialize ingestion component
        ingestion = DataIngestion()
        
        # Run the pipeline
        df, path, summary = ingestion.run()
        
        print(f"\n✅ Data Ingestion Test Successful!")
        print(f"   Records: {len(df)}")
        print(f"   Saved to: {path}")
        print(f"   Columns: {len(df.columns)}")
        
        if len(df) > 0:
            print(f"\nSample data (first 5 rows):")
            print(df.head())
            
    except Exception as e:
        print(f"\n❌ Data Ingestion Test Failed: {e}")
        sys.exit(1)