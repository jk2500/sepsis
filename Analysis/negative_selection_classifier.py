import os
import pandas as pd
import numpy as np
import concurrent.futures
from collections import Counter

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

# Utils from the project (assuming sepsis_analysis_utils.py is in PYTHONPATH or same directory)
try:
    from sepsis_analysis_utils import find_psv_files, get_column_headers
    from sepsis_data_augmentation_utils import augment_with_nsa_detectors
except ImportError:
    print("Warning: Could not import sepsis_analysis_utils or sepsis_data_augmentation_utils. Ensure it is in the correct path or PYTHONPATH.")
    def find_psv_files(directory, limit=None): return []
    def get_column_headers(file_path): return []

# --- Configuration (similar to vital_signs_classification.py) ---
DATA_DIRECTORY = './physionet.org/files/challenge-2019/1.0.0/training/training_setA/'
TARGET_COLUMN = 'SepsisLabel'
VITAL_SIGNS_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
TEST_SIZE = 0.25
RANDOM_STATE = 42
FILE_LIMIT = 1000 # For development speed, set to None for full dataset

# --- Negative Selection Algorithm Class ---
class NegativeSelectionAlgorithm:
    def __init__(self, detector_radius, num_detectors, max_trials_per_detector=1000, random_state=None):
        self.detector_radius = detector_radius
        self.num_detectors = num_detectors
        self.max_trials_per_detector = max_trials_per_detector
        self.random_state = random_state
        self.detectors_ = np.array([])
        self.feature_mins_ = None
        self.feature_maxs_ = None
        if self.random_state:
            np.random.seed(self.random_state)

    def _generate_candidate_detector(self):
        if self.feature_mins_ is None or self.feature_maxs_ is None:
            raise ValueError("Feature bounds not set. Call fit first.")
        return np.random.uniform(self.feature_mins_, self.feature_maxs_)

    def fit(self, X_train, y_train):
        print("Starting NSA training...")
        self_samples = X_train[y_train == 0]
        
        if self_samples.shape[0] == 0:
            print("Warning: No 'self' samples (y_train == 0) provided for training NSA. Cannot generate detectors.")
            self.detectors_ = np.array([])
            return

        # Determine feature bounds from the entire training set
        self.feature_mins_ = X_train.min(axis=0)
        self.feature_maxs_ = X_train.max(axis=0)

        generated_detectors = []
        attempts_total = 0

        for i in range(self.num_detectors):
            detector_found = False
            for trial in range(self.max_trials_per_detector):
                attempts_total += 1
                candidate_detector = self._generate_candidate_detector().reshape(1, -1)
                
                # Check if candidate matches any self sample
                distances_to_self = euclidean_distances(candidate_detector, self_samples)
                
                if np.all(distances_to_self > self.detector_radius):
                    generated_detectors.append(candidate_detector.flatten())
                    detector_found = True
                    if (i + 1) % (self.num_detectors // 10 if self.num_detectors >= 10 else 1) == 0:
                        print(f"  Generated {i+1}/{self.num_detectors} detectors...")
                    break 
            
            if not detector_found:
                print(f"  Warning: Could not generate detector {i+1}/{self.num_detectors} after {self.max_trials_per_detector} trials. Stopping detector generation.")
                break
        
        self.detectors_ = np.array(generated_detectors)
        if self.detectors_.shape[0] > 0:
            print(f"NSA training finished. Generated {self.detectors_.shape[0]} detectors. Total candidates tried: {attempts_total}.")
        else:
            print("NSA training finished. No detectors were generated.")


    def predict(self, X_to_classify):
        if self.detectors_.shape[0] == 0:
            print("Warning: No detectors available. Predicting all as 'self' (0).")
            return np.zeros(X_to_classify.shape[0], dtype=int)

        predictions = np.zeros(X_to_classify.shape[0], dtype=int)
        for i, sample in enumerate(X_to_classify):
            sample_reshaped = sample.reshape(1, -1)
            distances_to_detectors = euclidean_distances(sample_reshaped, self.detectors_)
            if np.any(distances_to_detectors <= self.detector_radius):
                predictions[i] = 1 # Classified as non-self (anomaly/sepsis)
        return predictions

# --- Data Loading and Preprocessing Functions (adapted from vital_signs_classification.py) ---
def _load_file_for_classification(file_path, relevant_headers):
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

def load_and_preprocess_data_for_nsa():
    print("Loading and preprocessing data for NSA...")
    psv_files = find_psv_files(DATA_DIRECTORY, limit=FILE_LIMIT)
    if not psv_files:
        print(f"No PSV files found in {DATA_DIRECTORY}. Exiting.")
        return None, None, None, None

    relevant_headers_for_loading = VITAL_SIGNS_FEATURES + [TARGET_COLUMN]
    all_data_dfs = []
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(_load_file_for_classification, p_file, relevant_headers_for_loading): p_file for p_file in psv_files}
        for future in concurrent.futures.as_completed(future_to_file):
            df_file = future.result()
            if df_file is not None and not df_file.empty: all_data_dfs.append(df_file)
    
    if not all_data_dfs:
        print("No data successfully loaded. Exiting.")
        return None, None, None, None
        
    full_df = pd.concat(all_data_dfs, ignore_index=True)
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    full_df[TARGET_COLUMN] = pd.to_numeric(full_df[TARGET_COLUMN], errors='coerce').astype('Int64')
    full_df.dropna(subset=[TARGET_COLUMN], inplace=True)

    if full_df.shape[0] < 10:
        print("Not enough data after cleaning target variable. Exiting.")
        return None, None, None, None

    X = full_df[VITAL_SIGNS_FEATURES]
    y = full_df[TARGET_COLUMN].astype(int)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Important: For NSA, typically no SMOTE on training data used to define 'self'
    # 'Self' is usually the normal, unaugmented data.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_imputed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    print(f"Data loaded. X_train_scaled: {X_train_scaled.shape}, y_train: {y_train.shape}, X_test_scaled: {X_test_scaled.shape}, y_test: {y_test.shape}")
    print(f"Training target distribution: {Counter(y_train)}")
    print(f"Test target distribution: {Counter(y_test)}")
    
    return X_train_scaled, y_train, X_test_scaled, y_test

# --- Main NSA Classification Function ---
def main_nsa_classification():
    print("Starting Negative Selection Algorithm Classification...")
    
    # Parameters for NSA (these will likely need tuning)
    detector_radius = 1  # Example: needs careful selection based on data scale
    num_detectors = 100000   # Example: number of detectors to attempt to generate
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data_for_nsa()

    if X_train is None:
        print("Failed to load data. Exiting NSA classification.")
        return

    nsa_classifier = NegativeSelectionAlgorithm(
        detector_radius=detector_radius,
        num_detectors=num_detectors,
        random_state=RANDOM_STATE
    )

    # Train NSA: provide all X_train and y_train, it will filter self samples internally
    nsa_classifier.fit(X_train, y_train)

    # Augment training data with NSA detectors using the utility function
    # X_train here is X_train_scaled from the preprocessing function
    X_train_augmented, y_train_augmented = augment_with_nsa_detectors(X_train, y_train, nsa_classifier)
    
    # The utility function handles printing, but you can add more if needed:
    # print(f"Shape of X_train after potential augmentation: {X_train_augmented.shape}")
    # print(f"Shape of y_train after potential augmentation: {y_train_augmented.shape}")
    # print(f"Target distribution in y_train_augmented: {Counter(y_train_augmented)}")

    # Predict on test set (original X_test)
    y_pred_nsa = nsa_classifier.predict(X_test)

    # Evaluate
    print("\n===== Negative Selection Algorithm Results =====")
    if nsa_classifier.detectors_.shape[0] == 0:
        print("No detectors were generated by NSA, so no meaningful evaluation possible.")
        print("This usually means the detector_radius is too large or self-space is too crowded.")
    else:
        acc = accuracy_score(y_test, y_pred_nsa)
        prec = precision_score(y_test, y_pred_nsa, zero_division=0)
        rec = recall_score(y_test, y_pred_nsa, zero_division=0)
        f1 = f1_score(y_test, y_pred_nsa, zero_division=0)
        # AUC might be tricky as basic NSA is binary; could adapt if predict_proba is implemented
        
        print(f"  Detectors Generated: {nsa_classifier.detectors_.shape[0]}")
        print(f"  Detector Radius: {detector_radius}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f} (Sepsis)")
        print(f"  Recall: {rec:.4f} (Sepsis)")
        print(f"  F1-score: {f1:.4f} (Sepsis)")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nsa)}")

    print("\nNSA classification script finished.")

if __name__ == '__main__':
    main_nsa_classification() 