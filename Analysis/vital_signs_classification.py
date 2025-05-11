import os
import pandas as pd
import numpy as np
import concurrent.futures
from collections import Counter

# Scikit-learn imports for preprocessing, model selection, models, and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Utils from the project
try:
    from sepsis_analysis_utils import find_psv_files, get_column_headers
    # Import for NSA augmentation
    from sepsis_data_augmentation_utils import augment_with_nsa_detectors
    from negative_selection_classifier import NegativeSelectionAlgorithm # Assuming it's in the same Analysis folder
except ImportError:
    print("Warning: Could not import sepsis_analysis_utils, sepsis_data_augmentation_utils, or NegativeSelectionAlgorithm. Ensure they are in the correct path or PYTHONPATH.")
    def find_psv_files(directory, limit=None): return []
    def get_column_headers(file_path): return []
    # Dummy NSA class if import fails, to allow script structure to be reasoned about
    class NegativeSelectionAlgorithm:
        def __init__(self, detector_radius, num_detectors, random_state=None):
            self.detectors_ = np.array([])
        def fit(self, X, y): print("Dummy NSA fit called."); pass
        def predict(self, X): return np.zeros(X.shape[0])
    def augment_with_nsa_detectors(X_train_scaled, y_train, nsa_model, verbose=True):
        print("Dummy augment_with_nsa_detectors called.")
        return X_train_scaled, y_train

# --- Configuration ---
DATA_DIRECTORY = './physionet.org/files/challenge-2019/1.0.0/training/training_setA/'
TARGET_COLUMN = 'SepsisLabel'
# Vital signs to be used as features
VITAL_SIGNS_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
# Note: EtCO2 is often unavailable; imputation will handle this.

TEST_SIZE = 0.25 # Using 25% for testing
RANDOM_STATE = 42
FILE_LIMIT = 1000 # Limit number of files for faster development/testing, set to None for all files

# NSA Parameters (can be tuned or moved to a config section)
NSA_DETECTOR_RADIUS = 1.0 # Example, needs tuning
NSA_NUM_DETECTORS = 10000 # Example, needs tuning

# --- Helper function to load and do initial prep for each file ---
def _load_file_for_classification(file_path, relevant_headers):
    """
    Loads a single PSV file, selects relevant columns, and converts to numeric.
    Relevant headers = VITAL_SIGNS_FEATURES + [TARGET_COLUMN]
    """
    try:
        with open(file_path, 'r') as f:
            try:
                actual_headers_in_file = f.readline().strip().split('|')
            except Exception:
                 return None 

        cols_to_read_from_file = [h for h in actual_headers_in_file if h in relevant_headers]
        if not cols_to_read_from_file:
            return pd.DataFrame(columns=relevant_headers, dtype=float)

        df = pd.read_csv(file_path, sep='|', header=0, usecols=cols_to_read_from_file, low_memory=False)
        
        for header in relevant_headers:
            if header not in df.columns:
                df[header] = np.nan
        df = df[relevant_headers] # Ensure correct order and all relevant columns

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return None 
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=relevant_headers, dtype=float)
    except Exception:
        return pd.DataFrame(columns=relevant_headers, dtype=float)

# --- Main classification function ---
def train_and_evaluate_vital_signs_classifiers():
    print(f"Starting vital signs classification...")
    print(f"Using features: {VITAL_SIGNS_FEATURES}")
    print(f"Target: {TARGET_COLUMN}")

    print(f"Finding PSV files (limit: {FILE_LIMIT})...")
    psv_files = find_psv_files(DATA_DIRECTORY, limit=FILE_LIMIT)
    
    if not psv_files:
        print(f"No PSV files found in {DATA_DIRECTORY}. Exiting.")
        return
    print(f"Found {len(psv_files)} PSV files to process.")

    # Determine headers to load: vital signs + target
    relevant_headers_for_loading = VITAL_SIGNS_FEATURES + [TARGET_COLUMN]

    print("Loading and concatenating data from files...")
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
            except Exception:
                pass # Error for specific file already handled by _load_file returning empty/None
            processed_files_count += 1
            if processed_files_count % (total_submitted // 10 if total_submitted >= 10 else 1) == 0 or processed_files_count == total_submitted:
                 print(f"File loading progress: {processed_files_count}/{total_submitted} futures completed...")

    if not all_data_dfs:
        print("No data successfully loaded. Exiting.")
        return
        
    full_df = pd.concat(all_data_dfs, ignore_index=True)
    print(f"Concatenated data shape: {full_df.shape}")

    if full_df.empty:
        print("Concatenated DataFrame is empty. Exiting.")
        return

    # --- Data Preprocessing ---
    print("\nPreprocessing data...")
    # Drop rows where the target is missing BEFORE defining X and y
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    full_df[TARGET_COLUMN] = pd.to_numeric(full_df[TARGET_COLUMN], errors='coerce').astype('Int64')
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True) # Drop again if coerce created NaNs

    if full_df.shape[0] < 10: # Arbitrary small number, ensure enough data after target drop
        print("Not enough data after cleaning target variable. Exiting.")
        return

    X = full_df[VITAL_SIGNS_FEATURES]
    y = full_df[TARGET_COLUMN].astype(int) # Ensure y is standard int for sklearn

    print(f"Shape of X (features): {X.shape}, Shape of y (target): {y.shape}")
    print(f"Target variable distribution: {Counter(y)}")

    if X.empty or y.empty or len(np.unique(y)) < 2:
        print("Not enough data or classes to proceed with classification. Exiting.")
        return

    print("Imputing missing values in features (mean strategy)...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=VITAL_SIGNS_FEATURES)

    print(f"Splitting data into train/test sets (test_size={TEST_SIZE})...")
    # X_train_raw/X_test_raw are not scaled yet
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_imputed_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Raw X_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
    print(f"Raw X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")
    print(f"Initial y_train distribution: {Counter(y_train)}")

    # Scale features (before NSA and SMOTE)
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw) # Scale test set with the same scaler
    print(f"X_train_scaled shape: {X_train_scaled.shape}")

    # --- NSA Training and Augmentation ---
    print("\n--- NSA Training and Data Augmentation ---")
    nsa_classifier = NegativeSelectionAlgorithm(
        detector_radius=NSA_DETECTOR_RADIUS,
        num_detectors=NSA_NUM_DETECTORS,
        random_state=RANDOM_STATE
    )
    print(f"Training NSA with radius: {NSA_DETECTOR_RADIUS}, num_detectors: {NSA_NUM_DETECTORS}...")
    # NSA's fit method expects scaled data and y_train
    nsa_classifier.fit(X_train_scaled, y_train) 

    X_train_augmented_scaled, y_train_augmented = augment_with_nsa_detectors(
        X_train_scaled, y_train, nsa_classifier, verbose=True
    )
    # y_train_augmented is the new y_train for SMOTE
    # X_train_augmented_scaled is the new X_train for SMOTE

    # Apply SMOTE to the (potentially NSA-augmented) training data
    print(f"\nApplying SMOTE to the training data (after potential NSA augmentation)...")
    print(f"Shape before SMOTE: X={X_train_augmented_scaled.shape}, y={y_train_augmented.shape}")
    print(f"Class distribution before SMOTE: {Counter(y_train_augmented)}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    # SMOTE expects X to be DataFrame or numpy array, y to be array-like.
    # X_train_augmented_scaled is already a numpy array from scaler/augmentation util.
    X_train_final, y_train_final = smote.fit_resample(X_train_augmented_scaled, y_train_augmented)
    
    print(f"Shape after SMOTE: X_train_final={X_train_final.shape}, y_train_final={y_train_final.shape}")
    print(f"Class distribution after SMOTE: {Counter(y_train_final)}")

    # --- Define Classifiers ---
    print("\nDefining classifiers...")
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced', n_jobs=-1),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_estimators=100, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear SVM": SVC(kernel='linear', probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
        "RBF SVM": SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, class_weight='balanced')
    }

    results = {}

    # --- Train, Predict, Evaluate Loop ---
    print("\nTraining and evaluating classifiers...")
    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        try:
            clf.fit(X_train_final, y_train_final) # Use final processed training data
            y_pred = clf.predict(X_test_scaled) # Use scaled test data
            
            # Default to 0 for metrics if y_pred_proba fails or not applicable
            auc_score_val = 0.0 
            if hasattr(clf, "predict_proba"):
                try:
                    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
                    auc_score_val = roc_auc_score(y_test, y_pred_proba)
                except Exception as e_auc:
                    print(f"    Could not calculate AUC for {name}: {e_auc}")
            else: # For models like KNN without probability=True for SVC without it (though we added it)
                 print(f"    {name} does not have predict_proba by default for AUC calculation or failed.")

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[name] = {
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
            # print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        except Exception as e:
            print(f"Error training/evaluating {name}: {e}")
            results[name] = {'Error': str(e)}

    print("\n\n===== Classification Results Summary (Vital Signs Only) =====")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        if 'Error' in metrics:
            print(f"  Error: {metrics['Error']}")
        else:
            for metric_name, value in metrics.items():
                print(f"  {metric_name:<12}: {value:.4f}")
    
    print("\nVital signs classification script finished.")

if __name__ == '__main__':
    train_and_evaluate_vital_signs_classifiers() 