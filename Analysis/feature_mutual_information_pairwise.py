import os
import pandas as pd
import numpy as np
from collections import defaultdict
import concurrent.futures

# Scikit-learn imports
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
# SimpleImputer is NOT needed for this pairwise approach

# Utils from the project
try:
    from sepsis_analysis_utils import find_psv_files, get_column_headers
except ImportError:
    print("Warning: Could not import sepsis_analysis_utils. Ensure it is in the correct path.")
    # Provide dummy functions if not found, so the rest of the script can be reasoned about
    def find_psv_files(directory, limit=None): return []
    def get_column_headers(file_path): return []


# --- Helper function to load and do initial prep for each file (same as in feature_mutual_information.py) ---
def _load_file_for_mi_pairwise(file_path, expected_headers):
    """
    Loads a single PSV file into a pandas DataFrame for pairwise MI.
    Selects only expected_headers and converts to numeric, coercing errors for NaNs.
    """
    try:
        with open(file_path, 'r') as f:
            try:
                actual_headers_in_file = f.readline().strip().split('|')
            except Exception: 
                 return None # For empty/corrupt small files

        cols_to_read_from_file = [h for h in actual_headers_in_file if h in expected_headers]
        if not cols_to_read_from_file:
            return pd.DataFrame(columns=expected_headers, dtype=float) # No relevant columns

        df = pd.read_csv(file_path, sep='|', header=0, usecols=cols_to_read_from_file, low_memory=False)
        
        for header in expected_headers:
            if header not in df.columns:
                df[header] = np.nan
        df = df[expected_headers]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return None 
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=expected_headers, dtype=float)
    except Exception as e:
        # print(f"Debug: Error in _load_file_for_mi_pairwise for {os.path.basename(file_path)}: {e}")
        return pd.DataFrame(columns=expected_headers, dtype=float)

# --- Main MI calculation function (Pairwise Complete Case) ---
def calculate_mutual_information_pairwise():
    data_directory = './physionet.org/files/challenge-2019/1.0.0/training/training_setB/'
    
    cols_to_ignore_as_features = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
    target_column = 'SepsisLabel'
    n_bins_discretization = 5 # Number of bins for discretizing features
    min_samples_for_mi = n_bins_discretization * 2 # Heuristic: need at least a few samples per bin on average

    print("Finding PSV files...")
    # Using limit=100 as per user's last setting in the other MI script for consistency in testing speed.
    # Change to None for all files.
    psv_files = find_psv_files(data_directory, limit=None) 
    
    if not psv_files:
        print(f"No PSV files found in {data_directory}. Exiting.")
        return
    print(f"Found {len(psv_files)} PSV files to process.")

    try:
        master_headers = get_column_headers(psv_files[0])
        if not master_headers:
            print(f"Could not read master headers from {psv_files[0]}. Exiting.")
            return
    except Exception as e:
        print(f"Error reading master headers from {psv_files[0]}: {e}. Exiting.")
        return

    print("Loading and concatenating data from files (this may take a while)...")
    all_data_dfs = []
    processed_files_count = 0
    files_with_load_errors = 0

    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(_load_file_for_mi_pairwise, p_file, master_headers): p_file for p_file in psv_files}
        total_submitted = len(future_to_file)
        print(f"Submitted {total_submitted} files for loading to {num_workers} worker threads.")

        for future in concurrent.futures.as_completed(future_to_file):
            file_path_key = future_to_file[future]
            try:
                df_file = future.result()
                if df_file is not None:
                    if not df_file.empty:
                        all_data_dfs.append(df_file)
                else:
                     files_with_load_errors +=1
            except Exception as e:
                files_with_load_errors +=1
            processed_files_count += 1
            if processed_files_count % (total_submitted // 20 if total_submitted >= 20 else 1) == 0 or processed_files_count == total_submitted:
                 print(f"File loading progress: {processed_files_count}/{total_submitted} futures completed...")
    
    if files_with_load_errors > 0:
        print(f"Warning: {files_with_load_errors} files may have had critical loading errors.")

    if not all_data_dfs:
        print("No data successfully loaded from any PSV files. Exiting.")
        return
        
    full_df = pd.concat(all_data_dfs, ignore_index=True)
    print(f"Concatenated data shape: {full_df.shape}")

    if full_df.empty:
        print("Concatenated DataFrame is empty. Exiting.")
        return
        
    print("Preparing target variable and identifying features...")
    if target_column not in full_df.columns:
        print(f"Target column '{target_column}' not found. Exiting.")
        return
        
    # Global preparation of target variable (SepsisLabel)
    full_df.dropna(subset=[target_column], inplace=True)
    full_df[target_column] = pd.to_numeric(full_df[target_column], errors='coerce').astype('Int64')
    full_df.dropna(subset=[target_column], inplace=True) # Drop again if coerce made NaNs

    if full_df.empty or full_df[target_column].nunique() < 2:
        print(f"Not enough valid data or target classes for '{target_column}' after initial NaN drop. Exiting.")
        return

    feature_columns = [col for col in master_headers if col != target_column and col not in cols_to_ignore_as_features and col in full_df.columns]
    if not feature_columns:
        print("No feature columns identified. Exiting.")
        return

    print(f"Found {len(feature_columns)} features to analyze pairwise.")
    mi_scores_dict = {}

    print("\nCalculating Mutual Information for each feature (pairwise complete case)...")
    for i, feature_name in enumerate(feature_columns):
        print(f"  Processing feature {i+1}/{len(feature_columns)}: {feature_name}")
        
        # Create a subset for the current feature and the target
        df_subset = full_df[[feature_name, target_column]].copy()
        df_subset.dropna(subset=[feature_name], inplace=True) # Key step: use only rows where this feature is present
        
        if df_subset.shape[0] < min_samples_for_mi: # Check if enough samples remain for this feature
            print(f"    Skipping '{feature_name}': Too few non-missing values ({df_subset.shape[0]}) after filtering.")
            mi_scores_dict[feature_name] = np.nan # Or 0.0, depending on how you want to treat it
            continue
        
        X_feature_series = df_subset[feature_name]
        y_labels_for_feature = df_subset[target_column].astype(int).to_numpy() # Corresponding labels for this subset

        if pd.Series(y_labels_for_feature).nunique() < 2:
            print(f"    Skipping '{feature_name}': Fewer than 2 unique target classes in its non-missing subset.")
            mi_scores_dict[feature_name] = np.nan
            continue

        # Discretize the current feature (X_feature_series)
        # Reshape X_feature_series to be a 2D array for KBinsDiscretizer
        X_feature_reshaped = X_feature_series.to_numpy().reshape(-1, 1)
        
        actual_n_bins = n_bins_discretization
        if X_feature_series.nunique() < 2:
            print(f"    Skipping '{feature_name}': Feature has only one unique value in its non-missing subset.")
            mi_scores_dict[feature_name] = 0.0 # No variation, so no MI with target
            continue
        if X_feature_series.nunique() < n_bins_discretization:
            actual_n_bins = X_feature_series.nunique()
            # print(f"    Note: '{feature_name}' has {actual_n_bins} unique values, using this for n_bins.")
        
        try:
            discretizer = KBinsDiscretizer(n_bins=max(2, actual_n_bins), encode='ordinal', strategy='quantile', subsample=min(200000, X_feature_reshaped.shape[0]), random_state=42)
            X_feature_discretized = discretizer.fit_transform(X_feature_reshaped)
        except ValueError as ve:
            # print(f"    Warning: Quantile discretization failed for '{feature_name}' ({ve}). Trying 'uniform'.")
            try:
                discretizer = KBinsDiscretizer(n_bins=max(2, actual_n_bins), encode='ordinal', strategy='uniform', subsample=min(200000, X_feature_reshaped.shape[0]), random_state=42)
                X_feature_discretized = discretizer.fit_transform(X_feature_reshaped)
            except Exception as e_uniform:
                print(f"    Error: Discretization failed for '{feature_name}' with both strategies: {e_uniform}. Assigning MI = NaN.")
                mi_scores_dict[feature_name] = np.nan
                continue
        
        # Calculate MI for this specific feature and its corresponding y labels
        try:
            current_mi = mutual_info_classif(X_feature_discretized, y_labels_for_feature, discrete_features=True, random_state=42)[0]
            mi_scores_dict[feature_name] = current_mi
        except Exception as mi_error:
            print(f"    Error calculating MI for '{feature_name}': {mi_error}. Assigning MI = NaN.")
            mi_scores_dict[feature_name] = np.nan

    # Display results
    mi_series = pd.Series(mi_scores_dict).sort_values(ascending=False)

    print("\n===== MUTUAL INFORMATION SCORES (Pairwise Complete Case) =====")
    print("(Higher score indicates more information shared with SepsisLabel)")
    print("(MI calculated for each feature using only its non-missing rows)")
    if mi_series.empty:
        print("No mutual information scores were calculated.")
    else:
        for feature, score in mi_series.items():
            print(f"{feature:<30}: {score if not pd.isna(score) else 'NaN'}") # Format NaN for printing
            
    print(f"\nNote: Features were discretized into up to {n_bins_discretization} bins using values present for each feature.")
    print("Features with too few samples or unique values after filtering non-missing entries may have NaN or 0 scores.")

if __name__ == '__main__':
    calculate_mutual_information_pairwise() 