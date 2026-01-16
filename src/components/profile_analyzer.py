from typing import Tuple, Dict, List, Any
import pandas as pd
import numpy as np
import itertools
import yaml
import sys
import os

from src.exception import CustomException
from src.logger import logging
from data.schemas import CustomerSegment, SegmentProfile

class ProfileAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize profile analyzer."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.profiling_config = self.config['profiling']
            self.segment_order = [
                CustomerSegment.LOW_HIGH,
                CustomerSegment.LOW_LOW,
                CustomerSegment.MEDIUM_HIGH,
                CustomerSegment.MEDIUM_LOW,
                CustomerSegment.HIGH
            ]
            
            logging.info("ProfileAnalyzer initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize ProfileAnalyzer: {e}")
            raise CustomException(e, sys)

    def calculate_odds_ratio(self, count: int, total_segment: int, 
                           pop_count: int, total_pop: int, eps: float = 1e-6) -> Tuple[float, float, float]:
        """Calculate odds ratio."""
        observed = count / total_segment if total_segment > 0 else 0.0
        expected = pop_count / total_pop if total_pop > 0 else 0.0
        or_value = observed / (expected + eps)
        return or_value, observed, expected

    def analyze_segment(self, segment_df: pd.DataFrame, segment: CustomerSegment,
                       population_df: pd.DataFrame, all_demographics: List[str]) -> List[Dict]:
        """Analyze a single segment for profile patterns."""
        try:
            seg_size = len(segment_df)
            total_size = len(population_df)
            profiles = []
            combo_id = 1
            
            # Find qualifying categories
            qualifying_categories = {}
            for demo in all_demographics:
                qualifying_categories[demo] = []
                cat_counts = segment_df[demo].value_counts(dropna=False)
                
                for value, count in cat_counts.items():
                    pop_count = population_df[demo].value_counts(dropna=False).get(value, 0)
                    or_value, obs, exp = self.calculate_odds_ratio(
                        count, seg_size, pop_count, total_size
                    )
                    
                    if or_value >= self.profiling_config['or_threshold']:
                        qualifying_categories[demo].append((value, or_value, obs, exp, count))
            
            # Get features with qualifying categories
            features_with_cats = [
                demo for demo, cats in qualifying_categories.items() 
                if cats
            ]
            
            if len(features_with_cats) < 2:
                return profiles
            
            # Try combinations from largest to smallest
            for r in range(len(features_with_cats), 1, -1):
                combo_records = []
                
                for feature_subset in itertools.combinations(features_with_cats, r):
                    category_options = [
                        [(demo, cat[0]) for cat in qualifying_categories[demo]]
                        for demo in feature_subset
                    ]
                    
                    for category_combo in itertools.product(*category_options):
                        # Create mask
                        mask = pd.Series(True, index=segment_df.index)
                        for demo, value in category_combo:
                            mask &= (segment_df[demo] == value)
                        match_count = int(mask.sum())
                        
                        # Apply coverage threshold
                        if match_count / seg_size < self.profiling_config['coverage_threshold']:
                            continue
                        
                        # Calculate population stats
                        pop_mask = pd.Series(True, index=population_df.index)
                        for demo, value in category_combo:
                            pop_mask &= (population_df[demo] == value)
                        pop_count = int(pop_mask.sum())
                        
                        or_value, obs, exp = self.calculate_odds_ratio(
                            match_count, seg_size, pop_count, total_size
                        )
                        
                        combo_records.append({
                            'segment': segment,
                            'combination_id': f"C{combo_id}",
                            'combination': " | ".join([f"{demo}: {value}" for demo, value in category_combo]),
                            'observed_proportion': obs,
                            'expected_proportion': exp,
                            'odds_ratio': or_value,
                            'count': match_count,
                            'coverage': match_count / seg_size
                        })
                        combo_id += 1
                
                if combo_records:
                    df_output = pd.DataFrame(combo_records)
                    df_output = df_output.sort_values(
                        by=["observed_proportion", "odds_ratio"],
                        ascending=[False, False]
                    ).head(self.profiling_config['max_combinations_per_segment'])
                    
                    profiles.extend(df_output.to_dict('records'))
                    break
            
            return profiles
            
        except Exception as e:
            logging.error(f"Segment analysis failed for {segment}: {e}")
            raise CustomException(e, sys)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run profile analysis on all segments."""
        try:
            logging.info("Starting profile analysis...")
            
            # Define demographics for profiling
            all_demographics = ['gender', 'marital_status', 'purpose', 
                               'employment_status', 'dw_channel_key', 
                               'age_category', 'income_bracket']
            
            all_profiles = []
            
            for segment in self.segment_order:
                logging.info(f"Analyzing segment: {segment.value}")
                
                segment_df = df[df['segment'] == segment].copy()
                if len(segment_df) == 0:
                    logging.warning(f"No customers in segment: {segment.value}")
                    continue
                
                profiles = self.analyze_segment(
                    segment_df, segment, df, all_demographics
                )
                
                all_profiles.extend(profiles)
                
                if profiles:
                    logging.info(f"  Found {len(profiles)} profile(s)")
                else:
                    logging.info(f"  No qualifying profiles found")
            
            # Convert to DataFrame
            profiles_df = pd.DataFrame(all_profiles)
            
            # Validate profiles
            validated_profiles = []
            for profile in all_profiles:
                try:
                    validated = SegmentProfile(**profile)
                    validated_profiles.append(validated.dict())
                except Exception as e:
                    logging.warning(f"Profile validation failed: {e}")
                    continue
            
            validated_df = pd.DataFrame(validated_profiles)
            logging.info(f"Generated {len(validated_df)} valid profiles")
            
            return validated_df
            
        except Exception as e:
            logging.error(f"Profile analysis pipeline failed: {e}")
            raise CustomException(e, sys)
