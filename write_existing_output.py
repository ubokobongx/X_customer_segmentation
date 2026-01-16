# main.py
import sys
import os
from dotenv import load_dotenv

from src.pipeline.train_pipeline import TrainPipeline
from src.exception import CustomException
from src.logger import logging

def main():
    """Main entry point for the customer segmentation pipeline."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variables
        required_vars = ['REDSHIFT_USER', 'REDSHIFT_PASSWORD', 
                        'PG_HOST', 'PG_DATABASE', 'PG_USER', 'PG_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logging.warning(f"Missing environment variables: {missing_vars}")
            logging.warning("Some database operations may fail")
        
        # Initialize and run training pipeline
        pipeline = TrainPipeline()
        result = pipeline.run()
        
        logging.info(f"Pipeline completed successfully!")
        logging.info(f"Run ID: {result['run_id']}")
        logging.info(f"Total customers: {result['stats']['total_customers']:,}")
        logging.info(f"Segments generated: {result['stats']['segments_generated']}")
        logging.info(f"Profiles generated: {result['stats']['profiles_generated']}")
        
        print("\n" + "="*80)
        print("CUSTOMER SEGMENTATION PIPELINE COMPLETED")
        print("="*80)
        print(f"Run ID: {result['run_id']}")
        print(f"Total customers processed: {result['stats']['total_customers']:,}")
        print(f"Segments generated: {result['stats']['segments_generated']}")
        print(f"Profiles generated: {result['stats']['profiles_generated']}")
        print(f"Artifacts saved in: artifacts/runs/{result['run_id']}/")
        print(f"Data written to PostgreSQL: ox.data_science.customer_segment")
        print(f"Profiles written to PostgreSQL: ox.data_science.segment_profile")
        print("="*80)
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()