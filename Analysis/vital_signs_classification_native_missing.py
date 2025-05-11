import os
import pandas as pd
import numpy as np
import concurrent.futures
from collections import Counter

# Scikit-learn imports for model selection and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance

# Gradient Boosting libraries with native missing value handling
try:
    import xgboost as xgb
except ImportError:
    print("Warning: xgboost not installed. XGBoost classifier will be a dummy.")
    class XGBClassifier: # Dummy class
        def __init__(self, random_state=None, use_label_encoder=False, eval_metric=None, n_jobs=-1): pass
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(X.shape[0])
        def predict_proba(self, X): return np.zeros((X.shape[0], 2))

try:
    import lightgbm as lgb
except ImportError:
    print("Warning: lightgbm not installed. LightGBM classifier will be a dummy.")
    class LGBMClassifier: # Dummy class
        def __init__(self, random_state=None, n_jobs=-1): pass # Add other common params if needed for dummy
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(X.shape[0])
        def predict_proba(self, X): return np.zeros((X.shape[0], 2))

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("Warning: catboost not installed. CatBoost classifier will be a dummy.")
    class CatBoostClassifier: # Dummy class
        def __init__(self, random_state=None, verbose=0): pass # Add other common params if needed
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(X.shape[0])
        def predict_proba(self, X): return np.zeros((X.shape[0], 2))

# Utils from the project
try:
    from sepsis_analysis_utils import find_psv_files
except ImportError:
    print("Warning: Could not import sepsis_analysis_utils. Ensure it is in the correct path or PYTHONPATH.")
    def find_psv_files(directory, limit=None): return []

# --- Configuration ---
DATA_DIRECTORY = './physionet.org/files/challenge-2019/1.0.0/training/training_setA/'
TARGET_COLUMN = 'SepsisLabel'
# VITAL_SIGNS_FEATURES will be replaced by dynamically discovered features
TEST_SIZE = 0.25
RANDOM_STATE = 42
FILE_LIMIT = None # Limit number of files for faster development/testing, set to None for all files

# --- Helper function to load and do initial prep for each file (same as in other scripts) ---
def _load_file_for_classification(file_path, relevant_headers):
    try:
        with open(file_path, 'r') as f:
            try:
                actual_headers_in_file = f.readline().strip().split('|')
            except Exception:
                 return None 
        cols_to_read_from_file = [h for h in actual_headers_in_file if h in relevant_headers]
        if not cols_to_read_from_file:
            return pd.DataFrame(columns=relevant_headers, dtype=float) # Return empty with correct columns if no relevant headers found
        df = pd.read_csv(file_path, sep='|', header=0, usecols=cols_to_read_from_file, low_memory=False)
        for header in relevant_headers: # Ensure all relevant columns exist, fill with NaN if not in file
            if header not in df.columns:
                df[header] = np.nan
        df = df[relevant_headers] # Ensure correct order and all relevant columns
        for col in df.columns:
            if col != TARGET_COLUMN: # Target column handled separately for to_numeric
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return None 
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=relevant_headers, dtype=float)
    except Exception as e:
        # print(f"Error loading file {file_path}: {e}") # Optional: for debugging file load errors
        return pd.DataFrame(columns=relevant_headers, dtype=float)

# --- Main classification function ---
def train_and_evaluate_native_missing_classifiers():
    print(f"Starting classification with native missing value handling for multiple models...")
    # Target remains the same, features will be discovered
    print(f"Target: {TARGET_COLUMN}")

    print(f"Finding PSV files (limit: {FILE_LIMIT})...")
    psv_files = find_psv_files(DATA_DIRECTORY, limit=FILE_LIMIT if FILE_LIMIT is not None else 0) # find_psv_files uses 0 for no limit if None
    
    if not psv_files:
        print(f"No PSV files found in {DATA_DIRECTORY}. Exiting.")
        return
    print(f"Found {len(psv_files)} PSV files to process.")

    # --- Step 1: Discover all unique column headers from the files ---
    def get_headers_from_file(file_path_local):
        try:
            with open(file_path_local, 'r') as f_local:
                return set(f_local.readline().strip().split('|'))
        except Exception:
            return set()

    print("\nDiscovering all feature columns from PSV files...")
    all_discovered_headers_set = set()
    # Limit scanning for headers if FILE_LIMIT is set, to speed up discovery
    # If FILE_LIMIT is high or None, this might scan many files.
    files_to_scan_for_headers = psv_files[:FILE_LIMIT] if FILE_LIMIT is not None else psv_files
    
    # Using ThreadPoolExecutor for potentially faster header discovery if many files
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_header_file = {executor.submit(get_headers_from_file, p_file): p_file for p_file in files_to_scan_for_headers}
        for future in concurrent.futures.as_completed(future_to_header_file):
            headers_in_file = future.result()
            if headers_in_file:
                all_discovered_headers_set.update(headers_in_file)

    if not all_discovered_headers_set:
        print("Could not discover any column headers from the files. Exiting.")
        return
    
    relevant_headers_for_loading = list(all_discovered_headers_set)
    print(f"Discovered {len(relevant_headers_for_loading)} unique column headers: {relevant_headers_for_loading}")
    # Ensure TARGET_COLUMN is included for loading if it was discovered, which it should be.
    if TARGET_COLUMN not in relevant_headers_for_loading:
        print(f"Warning: TARGET_COLUMN '{TARGET_COLUMN}' not found in discovered headers. Adding it for loading.")
        relevant_headers_for_loading.append(TARGET_COLUMN)
    # Note: _load_file_for_classification is designed to handle TARGET_COLUMN's specific processing.

    # --- Step 2: Load data using all discovered headers ---
    print("\nLoading and concatenating data from files using all discovered headers...")
    all_data_dfs = []
    processed_files_count = 0
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(_load_file_for_classification, p_file, relevant_headers_for_loading): p_file for p_file in psv_files}
        total_submitted = len(future_to_file)
        print(f"Submitted {total_submitted} files for loading to {num_workers} worker threads.")

        for future in concurrent.futures.as_completed(future_to_file):
            try:
                df_file = future.result()
                if df_file is not None and not df_file.empty:
                    all_data_dfs.append(df_file)
            except Exception as e:
                # file_path = future_to_file[future] # If you need to know which file failed
                # print(f"Error processing future for file: {e}") # Optional
                pass 
            processed_files_count += 1
            if processed_files_count % (total_submitted // 20 if total_submitted >= 20 else 1) == 0 or processed_files_count == total_submitted:
                 print(f"File loading progress: {processed_files_count}/{total_submitted} futures completed...")

    if not all_data_dfs:
        print("No data successfully loaded. Exiting.")
        return
        
    full_df = pd.concat(all_data_dfs, ignore_index=True)
    print(f"Concatenated data shape: {full_df.shape}")

    if full_df.empty:
        print("Concatenated DataFrame is empty. Exiting.")
        return

    # --- Data Preprocessing (Minimal: No Imputation, No Scaling for this version) ---
    print("\nPreprocessing data (minimal)...")
    # Drop rows where the target is missing
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    # Convert target to numeric and then to int. Coerce errors to NaN then drop again.
    full_df[TARGET_COLUMN] = pd.to_numeric(full_df[TARGET_COLUMN], errors='coerce')
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    full_df[TARGET_COLUMN] = full_df[TARGET_COLUMN].astype(int)


    if full_df.shape[0] < 10:
        print("Not enough data after cleaning target variable. Exiting.")
        return

    # --- Step 3: Define Features X using all columns from full_df except TARGET_COLUMN ---
    # And explicitly exclude 'Unit1' and 'Unit2'
    discovered_feature_columns = [col for col in full_df.columns if col != TARGET_COLUMN]
    
    # New: Define a list of manually selected important features
    SELECTED_IMPORTANT_FEATURES = [
        'ICULOS', 'Age', 'HospAdmTime', 'HR', 'DBP', 
        'SBP', 'Resp', 'O2Sat', 'MAP', 'Temp', 'Gender'
    ]
    
    # Filter discovered_feature_columns to only include those in SELECTED_IMPORTANT_FEATURES
    # and also ensure they are actually present in the loaded dataframe (full_df.columns)
    ALL_FEATURES = [col for col in SELECTED_IMPORTANT_FEATURES if col in full_df.columns]
    
    if not ALL_FEATURES:
        print(f"No selected important features ({SELECTED_IMPORTANT_FEATURES}) found in the loaded data columns. Exiting.")
        return
        
    X = full_df[ALL_FEATURES]
    y = full_df[TARGET_COLUMN]

    print(f"Shape of X (features: {len(ALL_FEATURES)} cols): {X.shape}, Shape of y (target): {y.shape}")
    print(f"Using features: {ALL_FEATURES}")
    print(f"Target variable distribution: {Counter(y)}")
    print(f"Missing values in X before split:\n{X.isnull().sum()}")


    if X.empty or y.empty or len(np.unique(y)) < 2:
        print("Not enough data or classes to proceed with classification. Exiting.")
        return

    # Split data - X will retain its NaNs here
    print(f"Splitting data into train/test sets (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Training target distribution: {Counter(y_train)}")
    print(f"Test target distribution: {Counter(y_test)}")
    print(f"Missing values in X_train:\n{X_train.isnull().sum()}")
    print(f"Missing values in X_test:\n{X_test.isnull().sum()}")


    # --- Calculate scale_pos_weight for imbalanced classification ---
    print("\nCalculating scale_pos_weight for imbalanced classification...")
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1 # Avoid division by zero
    print(f"  Negative samples in train: {neg_count}, Positive samples in train: {pos_count}")
    print(f"  Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

    # --- Define Classifiers --- 
    # These classifiers can handle NaNs natively.
    print("\nDefining classifiers with native missing value support and scale_pos_weight...")
    classifiers = {
        "XGBoost": xgb.XGBClassifier(
            random_state=RANDOM_STATE, 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight_value,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight_value,
            n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight_value,
            verbose=0
        )
    }

    results = {}

    # --- Train, Predict, Evaluate Loop ---
    print("\nTraining and evaluating classifiers...")
    for model_name, clf in classifiers.items():
        print(f"\n--- {model_name} ---")
        try:
            # Fit the model
            if model_name == "CatBoost":
                # CatBoost can use a list of feature names directly if X_train is a DataFrame
                # clf.fit(X_train, y_train, feature_names=X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None) # Incorrect parameter
                clf.fit(X_train, y_train) # If X_train is DataFrame, names are inferred.
            else:
                clf.fit(X_train, y_train) 

            y_pred = clf.predict(X_test)       
            
            auc_score_val = 0.0 # Default
            if hasattr(clf, "predict_proba"):
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                auc_score_val = roc_auc_score(y_test, y_pred_proba)
            else: 
                print(f"    {model_name} does not have predict_proba for AUC calculation.")

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[model_name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-score': f1,
                'AUC-ROC': auc_score_val
            }
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall: {rec:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  AUC-ROC: {auc_score_val:.4f}")
            cm = confusion_matrix(y_test, y_pred)
            print(f"  Confusion Matrix:\n{cm}")

        except Exception as e:
            print(f"Error training/evaluating or getting importances for {model_name}: {e}")
            results[model_name] = {'Error': str(e)}

    print("\n\n===== Classification Results Summary =====")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        if 'Error' in metrics:
            print(f"  Error: {metrics['Error']}")
        else:
            for metric_name, value in metrics.items():
                print(f"  {metric_name:<12}: {value:.4f}")
    
    print("\nClassification script (native missing handling for multiple models) finished.")

if __name__ == '__main__':
    train_and_evaluate_native_missing_classifiers() 