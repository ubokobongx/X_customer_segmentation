# src/utils/output_inspector.py
import pandas as pd
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def inspect_model_outputs(run_id: str = "20260115_092805"):
    """Inspect the model outputs from a specific run."""
    base_path = Path(f"artifacts/runs/{run_id}")
    
    print("=" * 60)
    print(f"INSPECTING MODEL OUTPUTS FOR RUN: {run_id}")
    print("=" * 60)
    
    # Check what files exist
    print("\nAvailable files:")
    for file_path in base_path.rglob("*"):
        if file_path.is_file():
            print(f"  - {file_path.relative_to(base_path)}")
    
    # Look for segmentation outputs
    potential_files = [
        base_path / "segmented_data.csv",
        base_path / "customer_segments.csv",
        base_path / "segmentation_results.csv",
        base_path / "profiles.csv",
        base_path / "segment_profiles.csv"
    ]
    
    for file_path in potential_files:
        if file_path.exists():
            print(f"\n{'='*40}")
            print(f"Found: {file_path.name}")
            print(f"{'='*40}")
            
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    print("\nFirst 3 rows:")
                    print(df.head(3).to_string())
                    
                    # Check for segment column
                    if 'segment' in df.columns:
                        print(f"\nSegment distribution:")
                        print(df['segment'].value_counts())
                
                elif file_path.suffix == '.json':
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"JSON structure keys: {list(data.keys())}")
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Also check the artifacts/data directory
    data_path = Path("artifacts/data")
    if data_path.exists():
        print(f"\n{'='*40}")
        print("Checking artifacts/data directory")
        print(f"{'='*40}")
        
        for data_file in data_path.glob("*.csv"):
            try:
                df = pd.read_csv(data_file)
                print(f"\n{data_file.name}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                
                # Look for processed data with segments
                if 'segment' in df.columns:
                    print(f"  Contains segmentation data!")
                    print(f"  Segment counts: {df['segment'].value_counts().to_dict()}")
                    
            except Exception as e:
                print(f"Error reading {data_file}: {e}")

if __name__ == "__main__":
    # Use the latest run ID from your logs
    inspect_model_outputs("20260115_092805")