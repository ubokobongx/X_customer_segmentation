# src/pipeline/train_pipeline.py
import pandas as pd
import yaml
import json
from typing import Dict, Any
import sys
import os
from datetime import datetime

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessor
from src.components.feature_engineering import FeatureEngineer
from src.components.segmenter import CustomerSegmenter
from src.components.profile_analyzer import ProfileAnalyzer
from src.components.postgres_writer import PostgreSQLWriter  # NEW IMPORT
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize training pipeline."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.artifacts_dir = "artifacts"
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.artifacts_dir, "runs", self.run_id)
            
            # Create run directory
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(os.path.join(self.run_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(self.run_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(self.run_dir, "reports"), exist_ok=True)
            
            logging.info(f"TrainPipeline initialized with run ID: {self.run_id}")
            
        except Exception as e:
            logging.error(f"Failed to initialize TrainPipeline: {e}")
            raise CustomException(e, sys)

    def save_artifacts(self, df: pd.DataFrame, profiles_df: pd.DataFrame, 
                      encoding_maps: Dict) -> Dict[str, str]:
        """Save all pipeline artifacts."""
        try:
            artifacts = {}
            
            # Save processed data
            data_path = os.path.join(self.run_dir, "data", "processed_data.parquet")
            df.to_parquet(data_path, index=False)
            artifacts['processed_data'] = data_path
            
            # Also save as CSV for easier inspection
            csv_path = os.path.join(self.run_dir, "data", "segmented_customers.csv")
            df.to_csv(csv_path, index=False)
            artifacts['segmented_customers_csv'] = csv_path
            
            # Save profiles
            profiles_path = os.path.join(self.run_dir, "models", "segment_profiles.parquet")
            profiles_df.to_parquet(profiles_path, index=False)
            artifacts['segment_profiles'] = profiles_path
            
            # Save profiles as CSV
            profiles_csv_path = os.path.join(self.run_dir, "models", "segment_profiles.csv")
            profiles_df.to_csv(profiles_csv_path, index=False)
            artifacts['segment_profiles_csv'] = profiles_csv_path
            
            # Save encoding maps
            encoding_path = os.path.join(self.run_dir, "models", "encoding_maps.json")
            with open(encoding_path, 'w') as f:
                json.dump(encoding_maps, f, indent=2)
            artifacts['encoding_maps'] = encoding_path
            
            # Save segmentation summary
            summary = {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'total_customers': len(df),
                'segmentation_distribution': df['segment'].value_counts().to_dict(),
                'profiles_generated': len(profiles_df),
                'artifacts': artifacts
            }
            
            summary_path = os.path.join(self.run_dir, "reports", "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            artifacts['summary'] = summary_path
            
            logging.info(f"Artifacts saved to: {self.run_dir}")
            return artifacts
            
        except Exception as e:
            logging.error(f"Failed to save artifacts: {e}")
            raise CustomException(e, sys)

    def generate_report(self, df: pd.DataFrame, profiles_df: pd.DataFrame) -> str:
        """Generate analysis report."""
        try:
            report_path = os.path.join(self.run_dir, "reports", "analysis_report.txt")
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("CUSTOMER SEGMENTATION ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Run ID: {self.run_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Total Customers: {len(df):,}\n\n")
                
                f.write("SEGMENT DISTRIBUTION:\n")
                f.write("-" * 40 + "\n")
                segment_counts = df['segment'].value_counts()
                for segment, count in segment_counts.items():
                    percentage = (count / len(df)) * 100
                    segment_name = segment.value if hasattr(segment, 'value') else str(segment)
                    f.write(f"{segment_name}: {count:,} ({percentage:.1f}%)\n")
                
                f.write("\nSEGMENT PROFILES:\n")
                f.write("-" * 40 + "\n")
                for segment in profiles_df['segment'].unique():
                    segment_profiles = profiles_df[profiles_df['segment'] == segment]
                    segment_name = segment.value if hasattr(segment, 'value') else str(segment)
                    f.write(f"\n{segment_name}:\n")
                    for _, profile in segment_profiles.iterrows():
                        f.write(f"  • {profile['combination']}\n")
                        f.write(f"    Coverage: {profile['coverage']:.1%}, OR: {profile['odds_ratio']:.2f}\n")
            
            logging.info(f"Report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"Failed to generate report: {e}")
            raise CustomException(e, sys)

    def run(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        try:
            logging.info("=" * 80)
            logging.info("STARTING CUSTOMER SEGMENTATION TRAINING PIPELINE")
            logging.info("=" * 80)
            
            # Step 1: Data Ingestion
            logging.info("\n[1/6] DATA INGESTION")
            ingestion = DataIngestion()
            raw_df, raw_data_path, ingestion_summary = ingestion.run()
            logging.info(f"Data ingestion summary: {ingestion_summary.get('total_records', 0)} records")
            
            # Step 2: Data Preprocessing
            logging.info("\n[2/6] DATA PREPROCESSING")
            preprocessor = DataPreprocessor()
            processed_df, encoding_maps = preprocessor.run(raw_df)
            
            # Step 3: Feature Engineering
            logging.info("\n[3/6] FEATURE ENGINEERING")
            feature_engineer = FeatureEngineer()
            feature_df = feature_engineer.run(processed_df)
            
            # Step 4: Customer Segmentation
            logging.info("\n[4/6] CUSTOMER SEGMENTATION")
            segmenter = CustomerSegmenter()
            segmented_df = segmenter.run(feature_df)
            
            # Step 5: Profile Analysis
            logging.info("\n[5/6] PROFILE ANALYSIS")
            analyzer = ProfileAnalyzer()
            profiles_df = analyzer.run(segmented_df)
            
            # Step 6: WRITE TO POSTGRESQL DATABASE (NEW STEP - UPDATED FOR NEW TABLES)
            logging.info("\n[6/6] WRITING TO POSTGRESQL DATABASE")
            postgres_writer = PostgreSQLWriter()
            
            # Write customer segments to customer_segment table (NEW TABLE NAME)
            postgres_writer.write_customer_segments(segmented_df)
            
            # Write segment profiles to segment_profile table (NEW TABLE NAME)
            postgres_writer.write_segment_profiles(profiles_df)
            
            postgres_writer.disconnect()
            logging.info("✅ PostgreSQL writing completed successfully")
            
            # Save artifacts locally
            artifacts = self.save_artifacts(segmented_df, profiles_df, encoding_maps)
            
            # Generate report
            report_path = self.generate_report(segmented_df, profiles_df)
            artifacts['report'] = report_path
            
            logging.info("=" * 80)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            
            return {
                'success': True,
                'run_id': self.run_id,
                'artifacts': artifacts,
                'stats': {
                    'total_customers': len(segmented_df),
                    'segments_generated': len(segmented_df['segment'].unique()),
                    'profiles_generated': len(profiles_df)
                }
            }
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise CustomException(e, sys)