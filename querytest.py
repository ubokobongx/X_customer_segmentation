# Create working copy with payment behavior features only
working_df = df[['customer_id', 'total_missed_installment', 'total_amount_overdue', 
                'tenor_in_months', 'total_loan_amount', 'maturity_dpd',
                'has_7plus_dpd', 'count_of_7plus_dpd', 'ontime_repayment_rate']].copy()

# Payment Behavior
working_df['missed_payment_ratio'] = working_df['total_missed_installment'] / (working_df['tenor_in_months'] + 1e-6)
working_df['overdue_utilization'] = working_df['total_amount_overdue'] / (working_df['total_loan_amount'] + 1e-6)

# Customer Value
working_df['monthly_loan_volume'] = working_df['total_loan_amount'] / (working_df['tenor_in_months'] + 1e-6)
working_df['repayment_efficiency'] = working_df['ontime_repayment_rate'] / 100  # Convert to 0-1 scale

def segment_customer(row):
    # More Lenient Risk Classification
    if (row['count_of_7plus_dpd'] == 0 and               
        row['missed_payment_ratio'] <= 0.4 and           
        row['maturity_dpd'] == 0):                       
        risk = 'Low'
    elif (row['count_of_7plus_dpd'] <= 2 and             
          row['missed_payment_ratio'] <= 0.7 and         
          row['maturity_dpd'] <= 4):                    
        risk = 'Medium'
    else:
        return 'High Risk'
    
    # More Inclusive Value Classification
    value = 'High Value' if (
        row['monthly_loan_volume'] >= working_df['monthly_loan_volume'].quantile(0.3) and  
        row['repayment_efficiency'] >= 0.2                                                 
    ) else 'Low Value'
    
    return f"{risk} Risk - {value}"

working_df['segment'] = working_df.apply(segment_customer, axis=1)

# Define segment order
segment_order = [
    'Low Risk - High Value',
    'Low Risk - Low Value',
    'Medium Risk - High Value', 
    'Medium Risk - Low Value',
    'High Risk'
]

# Calculate counts and percentages
segment_counts = working_df['segment'].value_counts()[segment_order]
percentages = (segment_counts / len(working_df)) * 100

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(segment_order, segment_counts, color=['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c'])

# Add labels
for bar, count, pct in zip(bars, segment_counts, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 5,  # Small offset above bar
             f'{count}\n({pct:.1f}%)',
             ha='center', 
             va='bottom',
             fontsize=10)

plt.title('Customer Segmentation Distribution', fontsize=14, pad=20)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.ylim(0, max(segment_counts) * 1.15)  # Add headroom for labels

# Remove top and right spines
sns.despine()

plt.tight_layout()
plt.show()

# 3. Cluster analysis setup
cluster_features = ['count_of_7plus_dpd', 'maturity_dpd', 
                   'missed_payment_ratio', 'overdue_utilization', 'ontime_repayment_rate']
all_demographics = demographic_features + ['age_category', 'income_bracket']

# Merge demographics
working_df = working_df.merge(
    df[['customer_id'] + all_demographics],
    on='customer_id',
    how='left'
)

# Standardize features
working_df[cluster_features] = StandardScaler().fit_transform(working_df[cluster_features])

# Create category mappings
category_maps = {
    feature: dict(enumerate(df[feature].astype('category').cat.categories))
    for feature in demographic_features
}

# 4. Population totals calculation
population_totals = {}
for demo in all_demographics:
    if demo in category_maps:
        population_totals[demo] = df[demo].map(category_maps[demo]).value_counts()
    else:
        population_totals[demo] = df[demo].value_counts()

# 5. Optimal cluster determination
def find_optimal_k(data, max_k=4):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    diffs = np.diff(inertias)
    diff_ratios = diffs[:-1] / diffs[1:]
    optimal_k = np.argmax(diff_ratios) + 2
    return min(max(2, optimal_k), 4)

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd

# Define function for stratified clustering (stratifying by key features)
def stratified_cluster(segment_data, cluster_features, n_clusters=2):
    # Split the segment data based on key features using stratified sampling
    # Here, we'll stratify on a feature that you choose, e.g., "loan amount" or "repayment behavior"
    
    # Example: Stratify based on 'loan_amount' and then apply KMeans
    # For simplicity, I'll stratify based on the cluster_features themselves
    X = segment_data[cluster_features]
    
    # Split the data using stratified sampling based on the cluster_features
    stratified_data = train_test_split(X, stratify=X['loan_amount'], test_size=0.5)[0]
    
    # Apply K-Means to the stratified data (you can choose your clustering method here)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    segment_data['sub_cluster'] = kmeans.fit_predict(stratified_data[cluster_features])
    
    return segment_data


segments_to_analyze = [
    'Low Risk - High Value',
    'Low Risk - Low Value',
    'Medium Risk - High Value',
    'Medium Risk - Low Value',
    'High Risk'
]

# Pre-calculate total population counts
population_totals = {}
for demo in all_demographics:
    if demo in category_maps:
        population_totals[demo] = df[demo].map(category_maps[demo]).value_counts()
    else:
        population_totals[demo] = df[demo].value_counts()

for segment in segments_to_analyze:
    # FIRST calculate segment rates for the ENTIRE segment (before clustering)
    segment_data = working_df[working_df['segment'] == segment].copy()
    
    # Skip clustering for small segments with less than 100 customers
    if len(segment_data) < 100:
        print(f"Skipping clustering for {segment} because it has fewer than 100 customers.")
        continue
    
    # Calculate segment rates once for the whole segment
    segment_rates = {}
    for demo in all_demographics:
        if demo in category_maps:
            demo_counts = segment_data[demo].map(category_maps[demo]).value_counts()
        else:
            demo_counts = segment_data[demo].value_counts()
        
        segment_rates[demo] = {
            value: count / population_totals[demo].get(value, 0.001)
            for value, count in demo_counts.items()
        }
    
    # Apply Stratified Clustering instead of K-Means (we assume n_clusters=2 for this example)
    segment_data = stratified_cluster(segment_data, cluster_features, n_clusters=2)
    
    print(f"\n{'='*50}")
    print(f"{segment.upper()} (Total: {len(segment_data):,})")
    print(f"{'='*50}")
    
    for cluster_num in range(2):  # We set n_clusters=2 for stratified clustering
        cluster_data = segment_data[segment_data['sub_cluster'] == cluster_num]
        
        print(f"\nSub-cluster {cluster_num+1} ({len(cluster_data):,} customers):")
        
        # Demographic analysis USING PRE-CALCULATED SEGMENT RATES and CALCULATING RELATIVE OVERREPRESENTATION
        for demo in all_demographics:
            print(f"\n{demo}:")
            
            # Get counts for this sub-cluster
            if demo in category_maps:
                counts = cluster_data[demo].map(category_maps[demo]).value_counts()
            else:
                counts = cluster_data[demo].value_counts()
            
            # Prepare table to calculate Relative Overrepresentation (OR)
            or_data = []
            for value, count in counts.items():
                observed_proportion = count / len(cluster_data)
                expected_proportion = population_totals[demo].get(value, 0) / len(df)
                
                relative_overrepresentation = observed_proportion / (expected_proportion + 0.001)  # Adding small value to avoid division by zero
                
                or_data.append({
                    'Value': value,
                    'Observed Proportion': round(observed_proportion, 4),
                    'Expected Proportion': round(expected_proportion, 4),
                    'Relative Overrepresentation (OR)': round(relative_overrepresentation, 2)
                })
            
            display_df = pd.DataFrame(or_data)
            display(display_df)
        
        # Behavioral profile remains unchanged
        print("\nBehavioral Profile:")
        display(cluster_data[cluster_features].mean().round(2))
