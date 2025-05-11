#!/usr/bin/env python
"""
Calculates and saves the mean and standard deviation statistics for features
in a sepsis patient dataset.
"""
import argparse
import os
import numpy as np
import pandas as pd
from glob import glob

# Attempt to import from model_MDN.py located in the same directory or sibling
try:
    from .model_MDN import FEATURE_COLUMNS, load_patient_file
except ImportError:
    print("Attempting fallback import for model_MDN components...")
    # This assumes model_MDN.py is in the python path or PYTHONPATH is set up.
    from model_MDN import FEATURE_COLUMNS, load_patient_file

def calculate_stats_from_files(patient_files):
    """
    Calculates mean and std for FEATURE_COLUMNS from a list of patient files.
    Args:
        patient_files (list): List of paths to patient data files.
    Returns:
        Tuple[np.ndarray, np.ndarray]: mean_stats, std_stats
    """
    all_feature_data = []
    print(f"Processing {len(patient_files)} patient files...")

    for i, file_path in enumerate(patient_files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(patient_files)} files...")
        try:
            df = load_patient_file(file_path)
            if not df.empty:
                # Ensure all FEATURE_COLUMNS are present, fill missing ones with NaN if necessary
                # This is to handle cases where a file might not have all columns explicitly,
                # though load_patient_file should ideally handle this by returning NaNs.
                for col in FEATURE_COLUMNS:
                    if col not in df.columns:
                        df[col] = np.nan
                all_feature_data.append(df[FEATURE_COLUMNS].values.astype(np.float32))
        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}. Skipping.")
            continue
    
    if not all_feature_data:
        print("No data loaded. Cannot calculate statistics.")
        return None, None

    # Concatenate all data into a single large NumPy array
    # This might be memory intensive for very large datasets.
    # For extremely large datasets, an incremental update approach might be needed,
    # but for typical research datasets, this should be acceptable.
    try:
        full_dataset_np = np.concatenate(all_feature_data, axis=0)
    except ValueError as ve:
        print(f"ValueError during concatenation: {ve}. This might happen if some files have zero rows after loading.")
        # Filter out empty arrays before concatenation
        all_feature_data_filtered = [arr for arr in all_feature_data if arr.shape[0] > 0]
        if not all_feature_data_filtered:
            print("No valid data after filtering empty arrays. Cannot calculate statistics.")
            return None, None
        full_dataset_np = np.concatenate(all_feature_data_filtered, axis=0)

    print(f"Calculating statistics over {full_dataset_np.shape[0]} total timesteps and {full_dataset_np.shape[1]} features.")

    # Calculate mean and std, ignoring NaNs
    mean_stats = np.nanmean(full_dataset_np, axis=0)
    std_stats = np.nanstd(full_dataset_np, axis=0)

    # Replace NaNs in std_stats (e.g., if a feature is all NaNs or single value) with 1.0 to avoid division by zero during normalization
    std_stats = np.where(np.isnan(std_stats), 1.0, std_stats)
    std_stats[std_stats == 0] = 1.0 # Also replace zero std with 1.0

    return mean_stats, std_stats

def main():
    parser = argparse.ArgumentParser(description="Calculate and save dataset feature statistics.")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing patient data files (PSV/CSV). Example: data/setA_full_train_test_split/train")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the calculated mean_stats.npy and std_stats.npy. Example: models/")
    parser.add_argument("--output_prefix", type=str, required=True, 
                        help="Prefix for the output files. Example: grud_mdn_setA")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    patient_files_psv = glob(os.path.join(args.data_dir, "*.psv"))
    patient_files_csv = glob(os.path.join(args.data_dir, "*.csv"))
    patient_files = patient_files_psv + patient_files_csv

    if not patient_files:
        print(f"No patient data files found in {args.data_dir}. Exiting.")
        return

    print(f"Found {len(patient_files)} patient files in {args.data_dir}.")

    mean_stats, std_stats = calculate_stats_from_files(patient_files)

    if mean_stats is not None and std_stats is not None:
        mean_filename = f"{args.output_prefix}_mean_stats.npy"
        std_filename = f"{args.output_prefix}_std_stats.npy"
        
        mean_output_path = os.path.join(args.output_dir, mean_filename)
        std_output_path = os.path.join(args.output_dir, std_filename)

        np.save(mean_output_path, mean_stats)
        np.save(std_output_path, std_stats)
        print(f"Mean statistics saved to: {mean_output_path}")
        print(f"Standard deviation statistics saved to: {std_output_path}")
        print("Mean values:", mean_stats)
        print("Std values:", std_stats)
    else:
        print("Failed to calculate statistics.")

if __name__ == "__main__":
    main() 