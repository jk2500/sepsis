#!/usr/bin/env python
"""
Analyzes a dataset of patient files to quantify the impact of active feature acquisition
on the uncertainty of a GRU-D Sepsis MDN model's predictions.

Usage (example):
    python sepsis_predictor_gru_d/analyze_dataset_active_info.py \
        --checkpoint_path /path/to/models/best_model_checkpoint.pt \
        --patient_data_dir /path/to/training_data/ \
        --mean_path /path/to/models/mean_stats.npy \
        --std_path /path/to/models/std_stats.npy \
        --output_csv_path active_information_analysis_results.csv \
        --top_k_uncertain_timesteps_per_patient 3
"""
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
import torch

# Attempt to import from sibling modules
try:
    from .model_MDN import SepsisGRUDMDN, FEATURE_COLUMNS, DEVICE, LABEL_COLUMN, TIME_COLUMN, load_patient_file
    from .inference_active import preprocess_single_patient_data, calculate_logit_variance, simulate_active_feature_acquisition
except ImportError as e:
    print(f"ImportError: {e}. Attempting fallback imports. Ensure scripts are in PYTHONPATH or run as part of a package.")
    # Fallback for direct execution if not run as part of a package
    from model_MDN import SepsisGRUDMDN, FEATURE_COLUMNS, DEVICE, LABEL_COLUMN, TIME_COLUMN, load_patient_file
    from inference_active import preprocess_single_patient_data, calculate_logit_variance, simulate_active_feature_acquisition

def analyze_dataset(
    checkpoint_path,
    patient_data_dir,
    mean_path,
    std_path,
    output_csv_path,
    uncertainty_threshold=None,
    top_k_uncertain_timesteps_per_patient=None,
    max_patients=None
):
    print(f"Using device: {DEVICE}")

    # 1. Load Normalization Statistics
    try:
        mean_stats = np.load(mean_path)
        std_stats = np.load(std_path)
        print(f"Loaded normalization statistics from {mean_path} and {std_path}")
    except Exception as e:
        print(f"Error loading normalization statistics: {e}. Exiting.")
        return

    # 2. Load Model from Checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path {checkpoint_path} does not exist. Exiting.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_args = checkpoint.get('args')
    if not model_args:
        print("Error: Checkpoint does not contain 'args' for model configuration. Exiting.")
        return

    num_original_features = len(FEATURE_COLUMNS)
    num_model_input_features = num_original_features
    if model_args.include_current_sepsis_label:
        num_model_input_features += 1

    model = SepsisGRUDMDN(
        input_size=num_model_input_features,
        hidden_size=model_args.hidden_size,
        num_gru_layers=model_args.num_gru_layers,
        dropout=model_args.dropout,
        num_mdn_components=model_args.num_mdn_components
    ).to(DEVICE)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}. Ensure model architecture matches checkpoint.")
        return
    model.eval()
    print(f"Model loaded successfully from {checkpoint_path}.")

    # 3. Find Patient Files
    patient_files = glob(os.path.join(patient_data_dir, "*.psv")) + glob(os.path.join(patient_data_dir, "*.csv"))
    if not patient_files:
        print(f"No patient files found in {patient_data_dir}. Exiting.")
        return
    if max_patients is not None:
        patient_files = patient_files[:max_patients]
    print(f"Found {len(patient_files)} patient files to analyze.")

    all_simulation_results = []
    files_processed_count = 0

    # 4. Process Each Patient File
    for i, patient_file_path in enumerate(patient_files):
        patient_id = os.path.basename(patient_file_path)
        print(f"\nProcessing patient {i+1}/{len(patient_files)}: {patient_id}")

        patient_data = preprocess_single_patient_data(
            patient_file_path,
            mean_stats,
            std_stats,
            max_seq_len=getattr(model_args, 'max_seq_len', None),
            include_current_sepsis_label=model_args.include_current_sepsis_label,
            feature_columns_list=FEATURE_COLUMNS,
            label_column_name=LABEL_COLUMN,
            time_column_name=TIME_COLUMN,
            num_original_features=num_original_features
        )

        if not patient_data or patient_data["length"] < 2: # Need at least 2 timesteps to make 1 prediction
            print(f"  Skipping patient {patient_id} due to insufficient data length after processing.")
            continue
        
        files_processed_count += 1
        x_p, m_p, delta_p, x_last_p, length_p = (
            patient_data["x"], patient_data["m"], patient_data["delta"], 
            patient_data["x_last"], patient_data["length"]
        )

        # Baseline full pass for this patient
        with torch.no_grad():
            pis_all, mus_all = model(
                x_p.unsqueeze(0).to(DEVICE), m_p.unsqueeze(0).to(DEVICE), 
                delta_p.unsqueeze(0).to(DEVICE), x_last_p.unsqueeze(0).to(DEVICE),
                torch.tensor([length_p], device=DEVICE, dtype=torch.long)
            )
        
        # Calculate baseline uncertainties for all valid prediction points
        baseline_uncertainties_for_patient = []
        # Predictions are for y_next[t], using inputs up to x[t], m[t], delta[t], x_last[t]
        # So, MDN output at index t corresponds to prediction for y_next[t]
        # Valid prediction indices for y_next[t] range from t=0 to t=length_p-2
        for t_idx in range(length_p - 1):
            pis_k_t = pis_all[0, t_idx, :]
            mus_k_t = mus_all[0, t_idx, :]
            uncertainty_t = calculate_logit_variance(pis_k_t, mus_k_t)
            baseline_uncertainties_for_patient.append((t_idx, uncertainty_t))
        
        if not baseline_uncertainties_for_patient:
            print(f"  No valid prediction timesteps for patient {patient_id}. Skipping.")
            continue

        # Identify target timesteps for active info simulation
        target_prediction_indices_for_analysis = []
        sorted_uncertainties = sorted(baseline_uncertainties_for_patient, key=lambda item: item[1], reverse=True)

        if uncertainty_threshold is not None:
            for t_idx, uncertainty in sorted_uncertainties:
                if uncertainty >= uncertainty_threshold:
                    target_prediction_indices_for_analysis.append(t_idx)
        elif top_k_uncertain_timesteps_per_patient is not None:
            for t_idx, _ in sorted_uncertainties[:top_k_uncertain_timesteps_per_patient]:
                target_prediction_indices_for_analysis.append(t_idx)
        else: # Default: analyze first valid timestep if no criteria, or could be all
            if baseline_uncertainties_for_patient: # Ensure there is at least one
                 target_prediction_indices_for_analysis.append(baseline_uncertainties_for_patient[0][0])
        
        if not target_prediction_indices_for_analysis:
            print(f"  No timesteps met analysis criteria for patient {patient_id}. Skipping active simulation for this patient.")
            continue
        
        print(f"  Analyzing {len(target_prediction_indices_for_analysis)} timesteps for patient {patient_id}: {target_prediction_indices_for_analysis}")

        for target_idx in target_prediction_indices_for_analysis:
            # simulate_active_feature_acquisition prints its own header, so it's fine here
            # It will use the x_p, m_p etc. which are single patient, full sequence tensors
            simulation_step_results = simulate_active_feature_acquisition(
                model, x_p, m_p, delta_p, x_last_p, length_p,
                target_idx, DEVICE, FEATURE_COLUMNS, num_original_features,
                model_args.include_current_sepsis_label, model_args.num_mdn_components,
                return_results=True
            )
            for res_dict in simulation_step_results:
                res_dict["patient_id"] = patient_id
                res_dict["timestep_idx_analyzed"] = target_idx # This is the index of the MDN output
                all_simulation_results.append(res_dict)
    
    if not files_processed_count:
        print("No patient files were successfully processed. Exiting.")
        return
        
    # 5. Aggregate and Save Results
    if not all_simulation_results:
        print("No simulation results collected. Exiting.")
        return

    results_df = pd.DataFrame(all_simulation_results)
    try:
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nAnalysis complete. Detailed results saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving results to CSV {output_csv_path}: {e}")
        print("Displaying partial results if available:")
        print(results_df.head())

    # Print some summary statistics
    if not results_df.empty:
        print("\nSummary Statistics:")
        # Average uncertainty reduction per feature
        avg_reduction_per_feature = results_df.groupby("feature_name")["uncertainty_reduction"].mean().sort_values(ascending=False)
        print("\nAverage Uncertainty Reduction per Feature (Top 10):")
        print(avg_reduction_per_feature.head(10))

        # Count how many times each feature was missing and analyzed
        feature_analysis_counts = results_df["feature_name"].value_counts()
        print("\nFrequency of Features Analyzed (Top 10):")
        print(feature_analysis_counts.head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run active information seeking analysis over a dataset.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--patient_data_dir", type=str, required=True, help="Directory with patient data files.")
    parser.add_argument("--mean_path", type=str, required=True, help="Path to mean_stats.npy.")
    parser.add_argument("--std_path", type=str, required=True, help="Path to std_stats.npy.")
    parser.add_argument("--output_csv_path", type=str, default="active_info_analysis.csv", help="Path to save CSV results.")
    parser.add_argument("--uncertainty_threshold", type=float, default=None, help="Logit variance threshold to trigger analysis.")
    parser.add_argument("--top_k_uncertain_timesteps_per_patient", type=int, default=None, help="Analyze top K most uncertain timesteps per patient.")
    parser.add_argument("--max_patients", type=int, default=None, help="Maximum number of patients to process (for testing).")

    cli_args = parser.parse_args()

    analyze_dataset(
        checkpoint_path=cli_args.checkpoint_path,
        patient_data_dir=cli_args.patient_data_dir,
        mean_path=cli_args.mean_path,
        std_path=cli_args.std_path,
        output_csv_path=cli_args.output_csv_path,
        uncertainty_threshold=cli_args.uncertainty_threshold,
        top_k_uncertain_timesteps_per_patient=cli_args.top_k_uncertain_timesteps_per_patient,
        max_patients=cli_args.max_patients
    ) 