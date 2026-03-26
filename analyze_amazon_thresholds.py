import pandas as pd
import os

# Configuration
DATASET_PATH = './dataset/Amazon_Books/Amazon_Books.inter'
THRESHOLDS = [250, 270, 280, 300, 320, 350, 370, 400, 450, 500]
OUTPUT_CSV = './amazon_threshold_analysis.csv'
ITEM_MIN_INTER = 5

def load_interactions(file_path):
    print(f"Loading data from {file_path}...")
    
    # Read just the header first to find column names
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = f.readline().strip()
    
    headers = header_line.split('\t')
    try:
        user_col = next(h for h in headers if h.startswith('user_id'))
        item_col = next(h for h in headers if h.startswith('item_id'))
    except StopIteration:
        print("Could not find user_id or item_id columns in header.")
        return None, None, None
    
    # Load the full dataset with error handling
    try:
        df = pd.read_csv(file_path, sep='\t', usecols=[user_col, item_col], encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decode failed, trying with errors='replace'...")
        df = pd.read_csv(file_path, sep='\t', usecols=[user_col, item_col], encoding='utf-8', encoding_errors='replace')
        
    return df, user_col, item_col

def calculate_sparsity(n_users, n_items, n_inter):
    if n_users == 0 or n_items == 0:
        return 1.0
    return 1 - (n_inter / (n_users * n_items))

def analyze_thresholds(df, user_col, item_col, thresholds):
    results = []
    
    # Calculate user interaction counts once
    print("Calculating user interaction counts...")
    user_counts = df[user_col].value_counts()
    
    for t in thresholds:
        print(f"Analyzing threshold {t}...")
        valid_users = user_counts[user_counts >= t].index
        
        # Filter interactions (User Filter Only)
        df_cut = df[df[user_col].isin(valid_users)]
        
        # Apply Item Filter
        item_counts = df_cut[item_col].value_counts()
        valid_items = item_counts[item_counts >= ITEM_MIN_INTER].index
        df_cut = df_cut[df_cut[item_col].isin(valid_items)]
        
        n_users = df_cut[user_col].nunique()
        n_items = df_cut[item_col].nunique()
        n_inter = len(df_cut)
        sparsity = calculate_sparsity(n_users, n_items, n_inter)
        
        print(f"  Users: {n_users}, Items: {n_items}, Interactions: {n_inter}, Sparsity: {sparsity:.6f}")
        
        results.append({
            'Threshold': t,
            'Users': n_users,
            'Items': n_items,
            'Interactions': n_inter,
            'Sparsity': sparsity
        })
        
    return pd.DataFrame(results)

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df, user_col, item_col = load_interactions(DATASET_PATH)
    
    if df is not None:
        results_df = analyze_thresholds(df, user_col, item_col, THRESHOLDS)
        
        print("\nFinal Results:")
        print(results_df)
        
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
