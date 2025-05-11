import os
import pandas as pd
import numpy as np
import concurrent.futures
from collections import Counter

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# UMAP and plotting imports
try:
    import umap
    import matplotlib.pyplot as plt
except ImportError:
    print("ImportError: `umap-learn` and `matplotlib` libraries are required for this script.")
    print("Please install them: pip install umap-learn matplotlib")
    # Exit if libraries are not found to prevent further errors
    import sys
    sys.exit(1)

# Utils from the project
try:
    from sepsis_analysis_utils import find_psv_files
except ImportError:
    print("Warning: Could not import sepsis_analysis_utils. Ensure it is in the correct path or PYTHONPATH.")
    def find_psv_files(directory, limit=None): return []

# --- Configuration ---
DATA_DIRECTORY = './physionet.org/files/challenge-2019/1.0.0/training/training_setA/'
TARGET_COLUMN = 'SepsisLabel'
VITAL_SIGNS_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
FILE_LIMIT = 1000  # Limit files for faster processing. Set to None for all data.
RANDOM_STATE = 42 # For UMAP reproducibility

# --- Data Loading and Preprocessing Functions (adapted) ---
def _load_file_for_umap(file_path, relevant_headers):
    try:
        with open(file_path, 'r') as f:
            try:
                actual_headers_in_file = f.readline().strip().split('|')
            except Exception: return None
        cols_to_read_from_file = [h for h in actual_headers_in_file if h in relevant_headers]
        if not cols_to_read_from_file: return pd.DataFrame(columns=relevant_headers, dtype=float)
        
        df = pd.read_csv(file_path, sep='|', header=0, usecols=cols_to_read_from_file, low_memory=False)
        for header in relevant_headers:
            if header not in df.columns: df[header] = np.nan
        df = df[relevant_headers]
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError: return None
    except pd.errors.EmptyDataError: return pd.DataFrame(columns=relevant_headers, dtype=float)
    except Exception: return pd.DataFrame(columns=relevant_headers, dtype=float)

def load_and_preprocess_data_for_umap():
    print(f"Loading and preprocessing data for UMAP (limit: {FILE_LIMIT} files)...")
    psv_files = find_psv_files(DATA_DIRECTORY, limit=FILE_LIMIT)
    if not psv_files:
        print(f"No PSV files found in {DATA_DIRECTORY}. Exiting.")
        return None, None

    relevant_headers_for_loading = VITAL_SIGNS_FEATURES + [TARGET_COLUMN]
    all_data_dfs = []
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(_load_file_for_umap, p_file, relevant_headers_for_loading): p_file for p_file in psv_files}
        processed_files_count = 0
        total_submitted = len(future_to_file)
        for future in concurrent.futures.as_completed(future_to_file):
            df_file = future.result()
            if df_file is not None and not df_file.empty: all_data_dfs.append(df_file)
            processed_files_count +=1
            if processed_files_count % (total_submitted // 10 if total_submitted >=10 else 1) == 0 or processed_files_count == total_submitted:
                print(f"  File loading progress: {processed_files_count}/{total_submitted} files processed...")

    if not all_data_dfs:
        print("No data successfully loaded. Exiting.")
        return None, None
        
    full_df = pd.concat(all_data_dfs, ignore_index=True)
    print(f"Initial concatenated data shape: {full_df.shape}")
    
    # Handle rows with NaN in target (important before separating X and y)
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    full_df[TARGET_COLUMN] = pd.to_numeric(full_df[TARGET_COLUMN], errors='coerce').astype('Int64')
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True) # Drop again if coerce created NaNs

    if full_df.empty or full_df.shape[0] < 10:
        print("Not enough data after cleaning target variable. Exiting.")
        return None, None

    X = full_df[VITAL_SIGNS_FEATURES]
    y = full_df[TARGET_COLUMN].astype(int)
    
    print(f"Shape of X (features): {X.shape}, Shape of y (target): {y.shape}")
    print(f"Target variable distribution: {Counter(y)}")

    # Impute missing values in features (X)
    print("Imputing missing values in features (mean strategy)...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    print("Data preparation for UMAP complete.")
    return X_scaled, y

# --- Main UMAP Visualization Function ---
def visualize_with_umap():
    print("Starting UMAP visualization for vital signs...")
    
    X_scaled, y = load_and_preprocess_data_for_umap()

    if X_scaled is None or y is None:
        print("Failed to load or preprocess data. Exiting UMAP visualization.")
        return

    if X_scaled.shape[0] < 2: # UMAP needs at least 2 samples
        print(f"Not enough samples ({X_scaled.shape[0]}) for UMAP. Exiting.")
        return
    
    # UMAP parameters (can be tuned)
    # n_neighbors controls how UMAP balances local versus global structure.
    # min_dist controls how tightly UMAP is allowed to pack points together.
    # n_components is 2 for 2D visualization.
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=5,  # Default is 15
        min_dist=0.1,    # Default is 0.1
        n_components=2,
        random_state=RANDOM_STATE,
        low_memory=True # Can help with larger datasets if memory becomes an issue
    )
    embedding = reducer.fit_transform(X_scaled)
    print(f"UMAP embedding shape: {embedding.shape}")

    print("Generating plot...")
    plt.figure(figsize=(12, 10))
    
    # Scatter plot, colored by SepsisLabel
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap='coolwarm', # 'coolwarm' or 'RdYlBu' are also good diverging cmaps
        s=5,             # marker size
        alpha=0.7        # marker transparency
    )
    
    plt.title(f'UMAP Projection of Vital Signs (Colored by SepsisLabel)\n{FILE_LIMIT if FILE_LIMIT else "All"} files, n_neighbors=15, min_dist=0.1', fontsize=15)
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    
    # Create a legend
    handles, _ = scatter.legend_elements(prop='colors', alpha=0.7)
    if len(np.unique(y)) == 2:
        legend_labels = ['Non-Sepsis (0)', 'Sepsis (1)']
        plt.legend(handles, legend_labels, title="SepsisLabel")
    elif len(np.unique(y)) == 1:
         legend_labels = [f'Class {np.unique(y)[0]}']
         plt.legend(handles, legend_labels, title="SepsisLabel (Only one class present)")
    else: # More than 2 classes or empty y (should not happen with current setup)
        plt.legend(handles, title="SepsisLabel")


    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot
    output_filename = 'umap_vital_signs_visualization.png'
    plt.savefig(output_filename)
    print(f"UMAP visualization saved to {output_filename}")
    
    # To prevent GUI windows from popping up if running in a headless environment,
    # we might avoid plt.show() or ensure backend is non-interactive.
    # For now, we'll assume saving is the primary goal.
    # plt.show() 

    print("\nUMAP visualization script finished.")

if __name__ == '__main__':
    visualize_with_umap() 