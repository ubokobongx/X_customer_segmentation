# inspect_profile_outputs_v2.py
import pandas as pd
import json
import sys
import os
from pathlib import Path

def inspect_profile_outputs():
    """Inspect the exact profile outputs from your model with focus on data types."""
    
    print("=" * 80)
    print("INSPECT PROFILE OUTPUTS - STRING customer_id VERSION")
    print("=" * 80)
    
    # Find the latest run
    artifacts_dir = Path("artifacts/runs")
    if not artifacts_dir.exists():
        print("❌ No artifacts directory found")
        return
    
    runs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()])
    if not runs:
        print("❌ No run directories found")
        return
    
    latest_run = runs[-1]
    print(f"📁 Latest run: {latest_run.name}")
    
    # Look for profile files
    profile_files = [
        latest_run / "models" / "segment_profiles.csv",
        latest_run / "models" / "segment_profiles.parquet",
        latest_run / "data" / "segmented_customers.csv",
        latest_run / "data" / "processed_data.parquet"
    ]
    
    # 1. Check segment_profiles.csv/parquet
    for file_path in profile_files[:2]:
        if file_path.exists():
            print(f"\n📊 Found profile file: {file_path}")
            try:
                if file_path.suffix == '.csv':
                    profiles_df = pd.read_csv(file_path)
                else:
                    profiles_df = pd.read_parquet(file_path)
                
                print(f"✅ Loaded {len(profiles_df)} profiles")
                
                # Check data types
                print(f"\n🔍 Data types in profiles:")
                for col in profiles_df.columns:
                    dtype = str(profiles_df[col].dtype)
                    sample = profiles_df[col].iloc[0] if len(profiles_df) > 0 else "N/A"
                    print(f"  • {col}: {dtype} (sample: {sample})")
                
                print(f"\n📋 Sample profiles (showing all):")
                for idx, row in profiles_df.iterrows():
                    print(f"\n--- Profile {idx + 1} ---")
                    for col in profiles_df.columns:
                        value = row[col]
                        value_type = type(value).__name__
                        print(f"  {col}: {value} ({value_type})")
                
                return profiles_df
                
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
    
    # 2. Check segmented_customers.csv for segment distribution
    for file_path in profile_files[2:]:
        if file_path.exists():
            print(f"\n📊 Found customer data file: {file_path}")
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_parquet(file_path)
                
                print(f"✅ Loaded {len(df)} customers")
                
                # Check customer_id data type
                if 'customer_id' in df.columns:
                    customer_id_dtype = str(df['customer_id'].dtype)
                    sample_ids = df['customer_id'].head(3).tolist()
                    print(f"\n🔍 customer_id analysis:")
                    print(f"  Data type: {customer_id_dtype}")
                    print(f"  Sample values: {sample_ids}")
                    print(f"  Unique values: {df['customer_id'].nunique()}")
                    print(f"  Is unique? {len(df) == df['customer_id'].nunique()}")
                    
                    # Check if it needs conversion to string
                    if customer_id_dtype.startswith('int'):
                        print(f"  ⚠️  Currently integer, will need conversion to string for PostgreSQL")
                
                if 'segment' in df.columns:
                    print(f"\n📈 Segment distribution:")
                    segment_counts = df['segment'].value_counts()
                    for segment, count in segment_counts.items():
                        percentage = (count / len(df)) * 100
                        segment_str = segment.value if hasattr(segment, 'value') else str(segment)
                        print(f"  • {segment_str}: {count:,} customers ({percentage:.1f}%)")
                
                # Check what other columns exist
                print(f"\n📋 Available columns and data types:")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    print(f"  • {col}: {dtype}")
                
                return df
                
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
    
    print("\n❌ Could not find any data files")
    return None

def analyze_combination_field(profiles_df):
    """Analyze the structure of combination field if it exists."""
    if 'combination' not in profiles_df.columns:
        return
    
    print("\n" + "=" * 80)
    print("ANALYSIS OF 'combination' FIELD STRUCTURE")
    print("=" * 80)
    
    # Get a sample combination
    sample_combinations = profiles_df['combination'].dropna().unique()[:3]
    
    print(f"\n📋 Found {len(sample_combinations)} unique combination patterns (showing first 3):")
    
    for i, combo in enumerate(sample_combinations, 1):
        print(f"\n--- Pattern {i} ---")
        print(f"Raw: '{combo}'")
        
        # Parse the combination
        if ' | ' in str(combo):
            parts = str(combo).split(' | ')
            print(f"Parsed into {len(parts)} parts:")
            
            for part in parts:
                if ': ' in part:
                    field, value = part.split(': ', 1)
                    print(f"  • Field: '{field.strip()}', Value: '{value.strip()}'")
                else:
                    print(f"  • Unknown format: '{part}'")
        else:
            print(f"  • No delimiter found, whole string: '{combo}'")
    
    # Check for demographic field patterns
    demographic_fields = ['marital_status', 'dw_channel_key', 'age_category', 
                         'income_bracket', 'employment_status', 'purpose', 
                         'gender', 'state', 'location']
    
    print(f"\n🔍 Looking for demographic fields in combinations:")
    found_fields = set()
    
    for combo in profiles_df['combination'].dropna().astype(str):
        for field in demographic_fields:
            if field in combo:
                found_fields.add(field)
    
    if found_fields:
        print(f"Found these demographic fields: {sorted(found_fields)}")
    else:
        print("No standard demographic fields found in combinations")

def generate_recommendation_mapping():
    """Generate the exact recommendation mapping you specified."""
    print("\n" + "=" * 80)
    print("RECOMMENDATION MAPPING (EXACT)")
    print("=" * 80)
    
    recommendations = {
        "Low Risk - High Value": "Prioritize & reward",
        "Low Risk - Low Value": "Grow value", 
        "Medium Risk - High Value": "Monitor risk, retain",
        "Medium Risk - Low Value": "Control exposure",
        "High Risk": "Restrict / red-flag"
    }
    
    print("\n📋 Exact recommendation mapping for new table:")
    print("+-----------------------------+---------------------+")
    print("| Segment                     | Recommendation      |")
    print("+-----------------------------+---------------------+")
    for segment, rec in recommendations.items():
        print(f"| {segment:28} | {rec:19} |")
    print("+-----------------------------+---------------------+")

def provide_next_steps():
    """Provide clear next steps."""
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR COMPLETE SOLUTION")
    print("=" * 80)
    print("\n1. 📊 Run this script and share the output with me")
    print("\n2. 📝 Provide demographic code mappings:")
    print("   - What does 'marital_status: 1' mean?")
    print("   - What does 'dw_channel_key: 1' mean? (Channel name)")
    print("   - What does 'employment_status: 2' mean?")
    print("   - What does 'purpose: 13' mean?")
    
    print("\n3. 🗃️ Create the new tables with:")
    print("   Table 1: customer_segment (customer_id VARCHAR PRIMARY KEY)")
    print("   Table 2: segment_profile (5 columns + timestamp)")
    
    print("\n4. 🔧 I'll create the updated PostgreSQL writer that:")
    print("   - Handles string customer_id")
    print("   - Uses your exact recommendation mapping")
    print("   - Properly formats demographic profiles")
    print("   - Writes to your new table structure")

if __name__ == "__main__":
    data_df = inspect_profile_outputs()
    
    if data_df is not None:
        # Check if it's profiles or customer data
        if 'combination' in data_df.columns:
            analyze_combination_field(data_df)
        elif 'customer_id' in data_df.columns:
            print(f"\n✅ Found customer data with {len(data_df)} records")
        
        generate_recommendation_mapping()
        provide_next_steps()
    else:
        print("\n❌ No data found. Please check your artifacts directory.")