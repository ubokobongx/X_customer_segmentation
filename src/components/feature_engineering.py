# src/components/feature_engineering.py
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any
import sys
import os

from src.exception import CustomException
from src.logger import logging
from data.schemas import ProcessedCustomerData

class FeatureEngineer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logging.info("FeatureEngineer initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize FeatureEngineer: {e}")
            raise CustomException(e, sys)

    def create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create payment behavior features."""
        try:
            logging.info("Creating payment behavior features...")
            
            # COMPREHENSIVE DEBUGGING
            logging.info("=" * 80)
            logging.info("🔍 FEATURE ENGINEERING INPUT ANALYSIS")
            logging.info("=" * 80)
            logging.info(f"DataFrame shape: {df.shape}")
            logging.info(f"Total columns: {len(df.columns)}")
            
            # Show all columns with their data types
            logging.info("Column details:")
            for i, col in enumerate(df.columns):
                unique_count = df[col].nunique() if col in df.columns else 0
                missing_count = df[col].isna().sum() if col in df.columns else 0
                logging.info(f"  {i+1:2d}. {col}: {df[col].dtype} (unique: {unique_count}, missing: {missing_count})")
            
            # Check for required columns
            required_columns = ['total_missed_installment', 'tenor_in_months', 
                              'total_amount_overdue', 'total_loan_amount',
                              'ontime_repayment_rate']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logging.error(f"❌ Missing required columns: {missing_columns}")
                
                # Check for alternative column names
                all_cols_lower = [str(col).lower() for col in df.columns]
                for missing in missing_columns:
                    similar = []
                    for actual_col in df.columns:
                        if missing.lower() in str(actual_col).lower() or str(actual_col).lower() in missing.lower():
                            similar.append(actual_col)
                    if similar:
                        logging.info(f"  Similar to '{missing}': {similar}")
                
                error_msg = f"Cannot create features without required columns: {missing_columns}"
                logging.error(error_msg)
                raise KeyError(error_msg)
            
            logging.info("✅ All required columns found!")
            
            # Show sample statistics for required columns
            for col in required_columns:
                if col in df.columns:
                    logging.info(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                               f"mean={df[col].mean():.2f}, missing={df[col].isna().sum()}")
            
            logging.info("=" * 80)
            
            # Create working copy
            working_df = df.copy()
            
            # Ensure numeric types
            for col in required_columns:
                if col in working_df.columns:
                    working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
            
            # Payment Behavior Features
            logging.info("Creating missed_payment_ratio...")
            working_df['missed_payment_ratio'] = (
                working_df['total_missed_installment'] / 
                (working_df['tenor_in_months'].replace(0, 1) + 1e-6)
            )
            
            logging.info("Creating overdue_utilization...")
            working_df['overdue_utilization'] = (
                working_df['total_amount_overdue'] / 
                (working_df['total_loan_amount'].replace(0, 1) + 1e-6)
            )
            
            # Customer Value Features
            logging.info("Creating monthly_loan_volume...")
            working_df['monthly_loan_volume'] = (
                working_df['total_loan_amount'] / 
                (working_df['tenor_in_months'].replace(0, 1) + 1e-6)
            )
            
            logging.info("Creating repayment_efficiency...")
            working_df['repayment_efficiency'] = (
                working_df['ontime_repayment_rate'] / 100  # Convert to 0-1 scale
            )
            
            # Clip values to reasonable ranges
            working_df['missed_payment_ratio'] = working_df['missed_payment_ratio'].clip(upper=10)
            working_df['overdue_utilization'] = working_df['overdue_utilization'].clip(upper=1)
            working_df['repayment_efficiency'] = working_df['repayment_efficiency'].clip(lower=0, upper=1)
            
            # Handle infinite values
            for col in ['missed_payment_ratio', 'overdue_utilization', 'monthly_loan_volume', 'repayment_efficiency']:
                if col in working_df.columns:
                    working_df[col] = working_df[col].replace([np.inf, -np.inf], np.nan)
                    # Fill NaN with median
                    median_val = working_df[col].median()
                    working_df[col] = working_df[col].fillna(median_val)
            
            # Log feature statistics
            logging.info("✅ Payment features created successfully")
            new_features = [col for col in working_df.columns if col not in df.columns]
            logging.info(f"   New features added: {new_features}")
            
            for feature in new_features:
                if feature in working_df.columns:
                    logging.info(f"   {feature}: mean={working_df[feature].mean():.4f}, "
                               f"std={working_df[feature].std():.4f}, "
                               f"min={working_df[feature].min():.4f}, "
                               f"max={working_df[feature].max():.4f}")
            
            logging.info(f"   Final shape: {working_df.shape}")
            logging.info("=" * 80)
            
            return working_df
            
        except Exception as e:
            logging.error(f"Failed to create payment features: {e}")
            # Add more context to the error
            logging.error(f"DataFrame info at error time:")
            if 'df' in locals():
                logging.error(f"  Shape: {df.shape}")
                logging.error(f"  Columns: {list(df.columns)}")
                logging.error(f"  First few rows columns: {list(df.columns)[:10] if len(df.columns) > 10 else list(df.columns)}")
            raise CustomException(e, sys)

    def create_risk_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-based segments."""
        try:
            logging.info("Creating risk segments...")
            
            working_df = df.copy()
            
            # Get risk thresholds from config
            risk_config = self.config['segmentation']['risk_thresholds']
            
            # Initialize all as high risk
            working_df['risk_level'] = 'High'
            
            # Apply risk thresholds
            # Low risk criteria
            low_risk_mask = (
                (working_df['count_14plus_dpd'] <= risk_config['low_risk']['max_count_14plus_dpd']) &
                (working_df['missed_payment_ratio'] <= risk_config['low_risk']['max_missed_payment_ratio']) &
                (working_df['maturity_dpd'] <= risk_config['low_risk']['max_maturity_dpd'])
            )
            
            # Medium risk criteria
            medium_risk_mask = (
                (working_df['count_14plus_dpd'] <= risk_config['medium_risk']['max_count_14plus_dpd']) &
                (working_df['missed_payment_ratio'] <= risk_config['medium_risk']['max_missed_payment_ratio']) &
                (working_df['maturity_dpd'] <= risk_config['medium_risk']['max_maturity_dpd']) &
                (~low_risk_mask)  # Not already low risk
            )
            
            # Apply masks
            working_df.loc[low_risk_mask, 'risk_level'] = 'Low'
            working_df.loc[medium_risk_mask, 'risk_level'] = 'Medium'
            
            logging.info(f"Risk segments created: {working_df['risk_level'].value_counts().to_dict()}")
            return working_df
            
        except Exception as e:
            logging.error(f"Failed to create risk segments: {e}")
            # If segmentation fails, add default risk level
            df['risk_level'] = 'Medium'
            return df

    def create_value_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create value-based segments."""
        try:
            logging.info("Creating value segments...")
            
            working_df = df.copy()
            
            # Get value thresholds from config
            value_config = self.config['segmentation']['value_thresholds']
            
            # Calculate percentiles
            monthly_loan_percentile = working_df['monthly_loan_volume'].quantile(
                value_config['monthly_loan_volume_percentile']
            )
            
            # Initialize all as low value
            working_df['value_level'] = 'Low Value'
            
            # High value criteria
            high_value_mask = (
                (working_df['monthly_loan_volume'] >= monthly_loan_percentile) &
                (working_df['repayment_efficiency'] >= value_config['repayment_efficiency_min'])
            )
            
            # Apply mask
            working_df.loc[high_value_mask, 'value_level'] = 'High Value'
            
            logging.info(f"Value segments created: {working_df['value_level'].value_counts().to_dict()}")
            return working_df
            
        except Exception as e:
            logging.error(f"Failed to create value segments: {e}")
            # If segmentation fails, add default value level
            df['value_level'] = 'Medium Value'
            return df

    def combine_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine risk and value segments."""
        try:
            logging.info("Combining risk and value segments...")
            
            working_df = df.copy()
            
            # Create combined segments
            segment_map = {
                ('Low', 'High Value'): 'Low Risk - High Value',
                ('Low', 'Low Value'): 'Low Risk - Low Value',
                ('Medium', 'High Value'): 'Medium Risk - High Value',
                ('Medium', 'Low Value'): 'Medium Risk - Low Value',
                ('High', 'High Value'): 'High Risk',
                ('High', 'Low Value'): 'High Risk'
            }
            
            # Apply mapping
            working_df['segment'] = working_df.apply(
                lambda row: segment_map.get((row['risk_level'], row['value_level']), 'Unknown'),
                axis=1
            )
            
            # Log segment distribution
            segment_counts = working_df['segment'].value_counts()
            logging.info(f"Segment distribution:")
            for segment, count in segment_counts.items():
                percentage = (count / len(working_df)) * 100
                logging.info(f"  {segment}: {count} ({percentage:.1f}%)")
            
            return working_df
            
        except Exception as e:
            logging.error(f"Failed to combine segments: {e}")
            # If combination fails, add default segment
            df['segment'] = 'Medium Risk - Medium Value'
            return df

    def save_features(self, df: pd.DataFrame, output_path: str) -> str:
        """Save engineered features."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as parquet for better performance
            save_path = output_path.replace('.csv', '.parquet') if output_path.endswith('.csv') else output_path
            
            try:
                df.to_parquet(save_path, index=False, compression='snappy')
                logging.info(f"Features saved as Parquet to: {save_path}")
                if os.path.exists(save_path):
                    logging.info(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
            except Exception as e:
                logging.warning(f"Failed to save as Parquet: {e}, falling back to CSV")
                save_path = output_path.replace('.parquet', '.csv') if output_path.endswith('.parquet') else output_path + '.csv'
                df.to_csv(save_path, index=False)
                logging.info(f"Features saved as CSV to: {save_path}")
            
            return save_path
            
        except Exception as e:
            logging.error(f"Failed to save features: {e}")
            raise CustomException(e, sys)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
        try:
            logging.info("=" * 80)
            logging.info("STARTING FEATURE ENGINEERING PIPELINE")
            logging.info("=" * 80)
            
            # Step 1: Create payment features
            logging.info("\n[1/4] CREATING PAYMENT FEATURES")
            df_with_features = self.create_payment_features(df)
            
            # Step 2: Create risk segments
            logging.info("\n[2/4] CREATING RISK SEGMENTS")
            df_with_risk = self.create_risk_segments(df_with_features)
            
            # Step 3: Create value segments
            logging.info("\n[3/4] CREATING VALUE SEGMENTS")
            df_with_value = self.create_value_segments(df_with_risk)
            
            # Step 4: Combine segments
            logging.info("\n[4/4] COMBINING SEGMENTS")
            df_with_segments = self.combine_segments(df_with_value)
            
            # Validate against schema
            logging.info("Validating features against schema...")
            records = df_with_segments.to_dict('records')
            validated_records = []
            validation_errors = 0
            
            for i, record in enumerate(records):
                try:
                    # Filter to only include schema fields
                    schema_fields = ProcessedCustomerData.__fields__
                    validated_record = {k: v for k, v in record.items() if k in schema_fields}
                    validated = ProcessedCustomerData(**validated_record)
                    validated_records.append(validated.dict())
                except Exception as e:
                    validation_errors += 1
                    if validation_errors <= 3:
                        logging.warning(f"Feature validation failed for record {i}: {e}")
            
            df_validated = pd.DataFrame(validated_records)
            
            logging.info("=" * 80)
            logging.info("FEATURE ENGINEERING COMPLETED")
            logging.info("=" * 80)
            logging.info(f"  Input records: {len(df)}")
            logging.info(f"  Output records: {len(df_validated)}")
            logging.info(f"  Validation errors: {validation_errors}")
            logging.info(f"  Total features created: {len(df_validated.columns)}")
            
            # Show final feature list
            logging.info("Final features:")
            for i, col in enumerate(df_validated.columns):
                unique_count = df_validated[col].nunique()
                logging.info(f"  {i+1:2d}. {col}: {df_validated[col].dtype} (unique: {unique_count})")
            
            return df_validated
            
        except Exception as e:
            logging.error(f"Feature engineering pipeline failed: {e}")
            raise CustomException(e, sys)