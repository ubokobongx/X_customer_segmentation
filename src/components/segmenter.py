# src/components/segmenter.py
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple
import sys
import os

from src.exception import CustomException
from src.logger import logging
from data.schemas import CustomerSegment, RiskLevel, ValueLevel

class CustomerSegmenter:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize customer segmenter."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.segmentation_config = self.config['segmentation']
            logging.info("CustomerSegmenter initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize CustomerSegmenter: {e}")
            raise CustomException(e, sys)

    def calculate_value_segment(self, row: pd.Series, df: pd.DataFrame) -> ValueLevel:
        """Calculate value segment for a customer."""
        try:
            monthly_loan_threshold = df['monthly_loan_volume'].quantile(
                self.segmentation_config['value_thresholds']['monthly_loan_volume_percentile']
            )
            repayment_threshold = self.segmentation_config['value_thresholds']['repayment_efficiency_min']
            
            if (row['monthly_loan_volume'] >= monthly_loan_threshold and 
                row['repayment_efficiency'] >= repayment_threshold):
                return ValueLevel.HIGH
            else:
                return ValueLevel.LOW
                
        except Exception as e:
            logging.warning(f"Value segmentation failed: {e}")
            return ValueLevel.LOW

    def segment_customer(self, row: pd.Series, df: pd.DataFrame) -> CustomerSegment:
        """Segment a single customer using rule-based logic."""
        try:
            risk_config = self.segmentation_config['risk_thresholds']
            
            # Risk Classification
            if (row['count_14plus_dpd'] <= risk_config['low_risk']['max_count_14plus_dpd'] and
                row['missed_payment_ratio'] <= risk_config['low_risk']['max_missed_payment_ratio'] and
                row['maturity_dpd'] <= risk_config['low_risk']['max_maturity_dpd']):
                risk = RiskLevel.LOW
                
            elif (row['count_14plus_dpd'] <= risk_config['medium_risk']['max_count_14plus_dpd'] and
                  row['missed_payment_ratio'] <= risk_config['medium_risk']['max_missed_payment_ratio'] and
                  row['maturity_dpd'] <= risk_config['medium_risk']['max_maturity_dpd']):
                risk = RiskLevel.MEDIUM
                
            else:
                return CustomerSegment.HIGH  # High Risk
            
            # Value Classification
            value = self.calculate_value_segment(row, df)
            
            # Combine risk and value
            if risk == RiskLevel.LOW and value == ValueLevel.HIGH:
                return CustomerSegment.LOW_HIGH
            elif risk == RiskLevel.LOW and value == ValueLevel.LOW:
                return CustomerSegment.LOW_LOW
            elif risk == RiskLevel.MEDIUM and value == ValueLevel.HIGH:
                return CustomerSegment.MEDIUM_HIGH
            elif risk == RiskLevel.MEDIUM and value == ValueLevel.LOW:
                return CustomerSegment.MEDIUM_LOW
            else:
                return CustomerSegment.HIGH
                
        except Exception as e:
            logging.warning(f"Customer segmentation failed: {e}")
            return CustomerSegment.HIGH

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run segmentation on all customers."""
        try:
            logging.info("Starting customer segmentation...")
            
            # Create working copy
            working_df = df.copy()
            
            # Apply segmentation
            working_df['segment'] = working_df.apply(
                lambda x: self.segment_customer(x, working_df), 
                axis=1
            )
            
            # Analyze segmentation results
            segment_counts = working_df['segment'].value_counts()
            percentages = (segment_counts / len(working_df)) * 100
            
            logging.info("Segmentation results:")
            for segment, count in segment_counts.items():
                pct = percentages[segment]
                logging.info(f"  {segment.value}: {count:,} customers ({pct:.1f}%)")
            
            logging.info(f"Segmented {len(working_df)} customers")
            return working_df
            
        except Exception as e:
            logging.error(f"Segmentation pipeline failed: {e}")
            raise CustomException(e, sys)