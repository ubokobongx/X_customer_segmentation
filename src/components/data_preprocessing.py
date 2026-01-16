# src/components/data_preprocessing.py
import pandas as pd
import numpy as np
import yaml
from typing import Tuple, Dict, Any
import sys
import os

from src.exception import CustomException
from src.logger import logging
from data.schemas import ProcessedCustomerData

class DataPreprocessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data preprocessor."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.preprocess_config = self.config['preprocessing']
            logging.info("DataPreprocessor initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize DataPreprocessor: {e}")
            raise CustomException(e, sys)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data."""
        try:
            logging.info("Cleaning data...")
            logging.info(f"Input shape: {df.shape}")
            
            # Create a copy
            df_clean = df.copy()
            
            # Clean gender field
            initial_count = len(df_clean)
            df_clean['gender'] = df_clean['gender'].fillna('Unknown')
            df_clean['gender'] = df_clean['gender'].apply(
                lambda x: 'Unknown' if str(x).strip().lower() == 'unknown' else str(x).strip().title()
            )
            
            # Remove rows with missing critical fields
            critical_fields = ['age', 'income', 'gender']
            df_clean = df_clean.dropna(subset=critical_fields)
            logging.info(f"Removed {initial_count - len(df_clean)} rows with missing critical fields")
            
            # Handle categorical missing values - fill with 'Others' or empty string
            cat_vars = ['employment_status', 'marital_status', 'state', 'location', 'purpose', 'dw_channel_key']
            for var in cat_vars:
                df_clean[var] = df_clean[var].fillna('')
                # Replace empty strings with 'Others' for categorical encoding
                df_clean[var] = df_clean[var].replace('', 'Others')
            
            # Apply age filter
            min_age = self.preprocess_config['min_age']
            max_age = self.preprocess_config['max_age']
            before_age_filter = len(df_clean)
            df_clean = df_clean[df_clean['age'] <= max_age]
            df_clean = df_clean[df_clean['age'] >= min_age]
            logging.info(f"Applied age filter ({min_age}-{max_age}): Removed {before_age_filter - len(df_clean)} rows")
            
            # Ensure numeric fields are properly typed
            numeric_fields = ['tenor_in_months', 'total_missed_installment', 'total_amount_overdue', 
                            'total_loan_amount', 'ontime_repayment_rate']
            for field in numeric_fields:
                if field in df_clean.columns:
                    df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
                    # Fill NaN with reasonable defaults
                    if field == 'tenor_in_months':
                        df_clean[field] = df_clean[field].fillna(1)
                    elif field == 'total_missed_installment':
                        df_clean[field] = df_clean[field].fillna(0)
                    elif field == 'total_amount_overdue':
                        df_clean[field] = df_clean[field].fillna(0)
                    elif field == 'total_loan_amount':
                        df_clean[field] = df_clean[field].fillna(df_clean[field].median())
                    elif field == 'ontime_repayment_rate':
                        df_clean[field] = df_clean[field].fillna(100)
            
            logging.info(f"Cleaned data shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logging.error(f"Data cleaning failed: {e}")
            raise CustomException(e, sys)

    def create_income_brackets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create income brackets."""
        try:
            logging.info("Creating income brackets...")
            
            bins = self.preprocess_config['income_bins']
            labels = self.preprocess_config['income_labels']
            
            # Ensure income is numeric
            df['income'] = pd.to_numeric(df['income'], errors='coerce')
            df['income'] = df['income'].fillna(df['income'].median())
            
            # Create bins
            df['income_bracket'] = pd.cut(
                df['income'],
                bins=bins + [float('inf')],
                labels=labels,
                right=False,
                include_lowest=True
            )
            
            # Fill any NaN brackets with the first bracket
            df['income_bracket'] = df['income_bracket'].fillna(labels[0])
            
            logging.info(f"Income brackets created: {df['income_bracket'].nunique()} categories")
            return df
            
        except Exception as e:
            logging.error(f"Failed to create income brackets: {e}")
            raise CustomException(e, sys)

    def create_age_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balanced age categories."""
        try:
            logging.info("Creating age categories...")
            
            min_age = self.preprocess_config['min_age']
            max_age = self.preprocess_config['max_age']
            target_groups = self.preprocess_config['age_groups']
            
            # Ensure age is within bounds
            df['age'] = df['age'].clip(lower=min_age, upper=max_age)
            
            # Create balanced age bins based on percentiles
            if len(df) >= target_groups:
                percentiles = np.linspace(0, 1, target_groups + 1)
                age_bins = [int(df['age'].quantile(q)) for q in percentiles]
                age_bins = [min_age] + age_bins[1:-1] + [max_age]
                
                # Ensure bins are unique and in order
                age_bins = sorted(list(set(age_bins)))
                
                # Create labels
                labels = []
                for i in range(len(age_bins)-1):
                    lower = int(age_bins[i])
                    upper = int(age_bins[i+1])
                    labels.append(f"{lower}-{upper}")
                
                # Create categories
                df['age_category'] = pd.cut(
                    df['age'],
                    bins=age_bins,
                    labels=labels,
                    right=True,
                    include_lowest=True
                )
            else:
                # If not enough data, create simple categories
                df['age_category'] = pd.cut(
                    df['age'],
                    bins=[min_age, 30, 40, 50, max_age],
                    labels=['18-30', '31-40', '41-50', '51+'],
                    right=True
                )
            
            # Fill any NaN categories
            if df['age_category'].isna().any():
                df['age_category'] = df['age_category'].fillna('18-30')
            
            logging.info(f"Created {df['age_category'].nunique()} age categories")
            return df
            
        except Exception as e:
            logging.error(f"Failed to create age categories: {e}")
            raise CustomException(e, sys)

    def encode_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables."""
        try:
            logging.info("Encoding categorical variables...")
            
            demographic_features = ['gender', 'marital_status', 'purpose', 
                                   'employment_status', 'dw_channel_key']
            
            # Store mapping for decoding if needed
            encoding_maps = {}
            
            for feature in demographic_features:
                if feature in df.columns:
                    # Fill NaN with 'Unknown' before encoding
                    df[feature] = df[feature].fillna('Unknown')
                    # Convert to string and then to category
                    df[feature] = df[feature].astype(str)
                    df[feature] = df[feature].astype('category').cat.codes
                    
                    # Store mapping
                    try:
                        encoding_maps[feature] = {
                            code: category for code, category in 
                            enumerate(df[feature].astype('category').cat.categories)
                        }
                    except:
                        encoding_maps[feature] = {}
                else:
                    logging.warning(f"Feature '{feature}' not found in DataFrame for encoding")
            
            return df, encoding_maps
            
        except Exception as e:
            logging.error(f"Categorical encoding failed: {e}")
            raise CustomException(e, sys)

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run complete preprocessing pipeline."""
        try:
            logging.info("=" * 60)
            logging.info("STARTING DATA PREPROCESSING PIPELINE")
            logging.info("=" * 60)
            
            # Step 1: Clean data
            logging.info("\n[1/4] CLEANING DATA")
            df_processed = self.clean_data(df)
            
            # Step 2: Create income brackets
            logging.info("\n[2/4] CREATING INCOME BRACKETS")
            df_processed = self.create_income_brackets(df_processed)
            
            # Step 3: Create age categories
            logging.info("\n[3/4] CREATING AGE CATEGORIES")
            df_processed = self.create_age_categories(df_processed)
            
            # Step 4: Encode categorical variables
            logging.info("\n[4/4] ENCODING CATEGORICAL VARIABLES")
            df_processed, encoding_maps = self.encode_categorical(df_processed)
            
            # Validate processed data
            logging.info("Validating processed data...")
            records = df_processed.to_dict('records')
            validated_records = []
            validation_errors = 0
            
            for i, record in enumerate(records):
                try:
                    # Filter to only include schema fields
                    validated_record = {k: v for k, v in record.items() 
                                      if k in ProcessedCustomerData.__fields__}
                    validated = ProcessedCustomerData(**validated_record)
                    validated_records.append(validated.dict())
                except Exception as e:
                    validation_errors += 1
                    if validation_errors <= 3:
                        logging.warning(f"Record {i} validation failed: {e}")
            
            df_validated = pd.DataFrame(validated_records)
            
            logging.info("=" * 60)
            logging.info("DATA PREPROCESSING COMPLETED")
            logging.info(f"  Input records: {len(df)}")
            logging.info(f"  Output records: {len(df_validated)}")
            logging.info(f"  Validation errors: {validation_errors}")
            logging.info(f"  Processed columns: {len(df_validated.columns)}")
            logging.info("=" * 60)
            
            # Show sample of processed data
            if len(df_validated) > 0:
                logging.info("Sample of processed data columns:")
                sample_cols = list(df_validated.columns)[:10]
                logging.info(f"  {sample_cols}")
            
            return df_validated, encoding_maps
            
        except Exception as e:
            logging.error(f"Data preprocessing pipeline failed: {e}")
            raise CustomException(e, sys)