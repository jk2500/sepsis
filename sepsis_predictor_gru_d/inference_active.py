#!/usr/bin/env python
"""
Performs inference using a pre-trained GRU-D Sepsis MDN model and
simulates active feature acquisition to assess impact on prediction uncertainty.

Usage (example):
    python sepsis_predictor_gru_d/inference_active.py \
        --checkpoint_path /path/to/your/models/best_model_checkpoint.pt \
        --patient_file_path /path/to/your/training_data/patient_XYZ.psv \
        --mean_path /path/to/your/models/mean_stats.npy \
        --std_path /path/to/your/models/std_stats.npy \
        --target_prediction_idx 0
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F # For F.binary_cross_entropy_with_logits in MDN loss context if ever needed here

# Attempt to import from model_MDN.py located in the same directory
try:
    from .model_MDN import SepsisGRUDMDN, FEATURE_COLUMNS, DEVICE, LABEL_COLUMN, TIME_COLUMN, load_patient_file
except ImportError:
    # Fallback for cases where the script might be run directly and '.' doesn't work as expected
    # This assumes model_MDN.py is in the python path or PYTHONPATH is set up.
    # For robust execution, ensure this script is part of a package or run with appropriate PYTHONPATH.
    print("Attempting fallback import for model_MDN components...")
    from model_MDN import SepsisGRUDMDN, FEATURE_COLUMNS, DEVICE, LABEL_COLUMN, TIME_COLUMN, load_patient_file

# ---------------------------
# Active Inference Utilities (copied from earlier design)
# ---------------------------
def calculate_predictive_entropy(pis_k, mus_k):
    """Calculates predictive entropy (in bits) from MDN parameters for a binary outcome."""
    # pis_k: tensor of shape (num_mdn_components), on some device
    # mus_k: tensor of shape (num_mdn_components), on same device
    # Assumes mus_k are logits.
    device = pis_k.device
    dtype = pis_k.dtype

    probs_k = torch.sigmoid(mus_k)  # Probabilities of sepsis for each component
    mean_prob = torch.sum(pis_k * probs_k)  # Expected probability of sepsis

    # Clamp mean_prob to avoid log(0) or log(1) issues leading to NaN
    eps = torch.tensor(1e-7, device=device, dtype=dtype)
    mean_prob_clamped = torch.clamp(mean_prob, min=eps, max=1.0 - eps)

    # Calculate entropy using log base 2 for bits
    entropy = - (mean_prob_clamped * torch.log2(mean_prob_clamped) + \
                 (1.0 - mean_prob_clamped) * torch.log2(1.0 - mean_prob_clamped))
    
    return entropy.item()

def simulate_active_feature_acquisition(
    model,
    x_patient_seq, m_patient_seq, delta_patient_seq, x_last_patient_seq, patient_length,
    target_prediction_idx, # Index in the MDN output sequence, e.g., 0 for predicting y_next[0]
    device,
    feature_column_names, # Original list of feature names
    num_original_features,
    include_current_sepsis_label_in_model, # From loaded model args
    num_mdn_components, # From loaded model args
    return_results=False # New argument
):
    """
    Simulates observing missing features one by one using sample-based marginalization
    and measures the change in predictive entropy (Information Gain) for the corresponding prediction.
    If return_results is True, returns a list of dicts instead of printing details.
    """
    model.eval()
    simulation_outputs = [] # To store results if return_results is True

    # Placeholder for sampling: Use mean, mean+std, mean-std in normalized space
    # K_SAMPLES_FOR_EXPECTATION = 3
    # These are normalized values, as the model expects normalized inputs.
    # 0.0 is mean, 1.0 is approx +1 std, -1.0 is approx -1 std in normalized space.
    hypothetical_normalized_samples = torch.tensor([0.0, 1.0, -1.0], device=device, dtype=x_patient_seq.dtype)


    input_timestep_idx = target_prediction_idx # GRU-D output at t corresponds to input at t

    if not (0 <= input_timestep_idx < patient_length -1): # Need at least one step to predict for (y_next[t] means actual y[t+1])
        print(f"  Target prediction index {target_prediction_idx} (input step {input_timestep_idx}) is not valid for patient length {patient_length}. Min length 2 required for predicting a future step. Skipping active info seeking.")
        if return_results:
            return []
        return

    print(f"\n--- Active Information Seeking Simulation (for MDN output at index {target_prediction_idx}) ---")
    print(f"--- Considering inputs at sequence index {input_timestep_idx} to predict y_next[{target_prediction_idx}] ---")

    x_batch = x_patient_seq.unsqueeze(0).to(device)
    m_batch = m_patient_seq.unsqueeze(0).to(device)
    delta_batch = delta_patient_seq.unsqueeze(0).to(device)
    x_last_batch = x_last_patient_seq.unsqueeze(0).to(device)
    lengths_batch = torch.tensor([patient_length], device=device, dtype=torch.long)

    with torch.no_grad():
        pis_baseline_all_steps, mus_baseline_all_steps = model(x_batch, m_batch, delta_batch, x_last_batch, lengths_batch)

    pis_k_baseline = pis_baseline_all_steps[0, target_prediction_idx, :]
    mus_k_baseline = mus_baseline_all_steps[0, target_prediction_idx, :]
    
    # Calculate baseline mean probability and entropy
    probs_k_baseline = torch.sigmoid(mus_k_baseline)
    mean_prob_baseline = torch.sum(pis_k_baseline * probs_k_baseline).item()
    baseline_entropy = calculate_predictive_entropy(pis_k_baseline, mus_k_baseline)
    
    print(f"  Baseline P(Sepsis=1) for y_next[{target_prediction_idx}]: {mean_prob_baseline:.4f}")
    print(f"  Baseline Predictive Entropy for y_next[{target_prediction_idx}]: {baseline_entropy:.4f} bits")

    missing_feature_indices_at_input_step = torch.where(m_patient_seq[input_timestep_idx, :num_original_features] == 0)[0]
    if include_current_sepsis_label_in_model and m_patient_seq[input_timestep_idx, num_original_features] == 0:
        # SepsisLabel_t is the last feature if included
        sepsis_label_model_idx = num_original_features 
        missing_feature_indices_at_input_step = torch.cat((missing_feature_indices_at_input_step, torch.tensor([sepsis_label_model_idx], device=missing_feature_indices_at_input_step.device)))

    if missing_feature_indices_at_input_step.numel() == 0:
        print(f"  No missing features at input timestep {input_timestep_idx} to simulate acquiring.")
        if return_results:
            return []
        return
    
    print(f"  Missing features at input timestep {input_timestep_idx} (model input indices): {missing_feature_indices_at_input_step.tolist()}")

    for feature_model_idx_tensor in missing_feature_indices_at_input_step:
        feature_model_idx = feature_model_idx_tensor.item()
        feature_name = "Unknown"
        if feature_model_idx < num_original_features:
            feature_name = feature_column_names[feature_model_idx]
        elif include_current_sepsis_label_in_model and feature_model_idx == num_original_features:
            feature_name = "SepsisLabel_t (as input feature)"
        else:
            if not return_results:
                 print(f"    Warning: Could not map model feature index {feature_model_idx} to a name. Skipping.")
            continue
        
        sum_entropy_after_acquisition = 0.0
        
        for sample_val_tensor in hypothetical_normalized_samples:
            sample_val = sample_val_tensor.item()

            x_mod_seq = x_patient_seq.clone()
            m_mod_seq = m_patient_seq.clone()
            delta_mod_seq = delta_patient_seq.clone() # Ensure delta is fresh for each modification

            # Impute the hypothetical sampled value
            x_mod_seq[input_timestep_idx, feature_model_idx] = sample_val 
            m_mod_seq[input_timestep_idx, feature_model_idx] = 1.0      # Mark as observed
            delta_mod_seq[input_timestep_idx, feature_model_idx] = 0.0  # Time since last observation is 0

            x_mod_batch = x_mod_seq.unsqueeze(0).to(device)
            m_mod_batch = m_mod_seq.unsqueeze(0).to(device)
            delta_mod_batch = delta_mod_seq.unsqueeze(0).to(device)
            # x_last_batch and lengths_batch remain the same as baseline call

            with torch.no_grad():
                pis_modified_all_steps, mus_modified_all_steps = model(
                    x_mod_batch, m_mod_batch, delta_mod_batch, x_last_batch, lengths_batch
                )
            
            pis_k_modified = pis_modified_all_steps[0, target_prediction_idx, :]
            mus_k_modified = mus_modified_all_steps[0, target_prediction_idx, :]
            
            entropy_for_sample = calculate_predictive_entropy(pis_k_modified, mus_k_modified)
            sum_entropy_after_acquisition += entropy_for_sample
        
        expected_entropy_after_acquisition = sum_entropy_after_acquisition / len(hypothetical_normalized_samples)
        information_gain = baseline_entropy - expected_entropy_after_acquisition

        if return_results:
            simulation_outputs.append({
                "feature_name": feature_name,
                "feature_model_idx": feature_model_idx,
                "baseline_entropy_at_timestep": baseline_entropy,
                "expected_entropy_after_acquiring_feature": expected_entropy_after_acquisition,
                "information_gain": information_gain
            })
        else:
            print(f"    If '{feature_name}' (model_idx {feature_model_idx}) was acquired (simulated with {len(hypothetical_normalized_samples)} samples):")
            print(f"      Expected Entropy After Acquisition: {expected_entropy_after_acquisition:.4f} bits, Information Gain: {information_gain:.4f} bits")
    
    if not return_results:
        print("--- End of Active Information Seeking Simulation ---")
    
    if return_results:
        return simulation_outputs

# ---------------------------
# Data Preprocessing for Single Patient
# ---------------------------
def preprocess_single_patient_data(file_path, mean_stats, std_stats, max_seq_len, 
                                   include_current_sepsis_label, 
                                   feature_columns_list, label_column_name, time_column_name,
                                   num_original_features):
    """
    Loads and preprocesses a single patient file for inference.
    Adapted from SepsisDataset._get_single_item_data_from_file.
    Returns a dictionary of tensors or None if processing fails.
    """
    try:
        df = load_patient_file(file_path) # load_patient_file is imported
    except Exception as e:
        print(f"Error loading patient file {file_path}: {e}")
        return None

    if max_seq_len is not None:
        df = df.iloc[:max_seq_len]
            
    x_original = df[feature_columns_list].values.astype(np.float32)
    y_sepsis_labels_current_t = df[label_column_name].values.astype(np.float32) 

    seq_len = x_original.shape[0]
    if seq_len == 0: 
        print(f"Warning: Patient file {file_path} resulted in zero sequence length after potential truncation.")
        # Determine num_input_features based on args
        num_input_features = num_original_features + (1 if include_current_sepsis_label else 0)
        return {
            "x": torch.empty((0, num_input_features), dtype=torch.float32),
            "m": torch.empty((0, num_input_features), dtype=torch.float32),
            "delta": torch.empty((0, num_input_features), dtype=torch.float32),
            "x_last": torch.empty((0, num_input_features), dtype=torch.float32),
            "y_current": torch.empty(0, dtype=torch.float32),
            "y_next": torch.empty(0, dtype=torch.float32),
            "length": 0,
            "patient_df": df 
        }

    m_original = (~np.isnan(x_original)).astype(np.float32)
    
    # Ensure mean_stats and std_stats are numpy arrays and match num_original_features
    if not (isinstance(mean_stats, np.ndarray) and mean_stats.shape == (num_original_features,) and
            isinstance(std_stats, np.ndarray) and std_stats.shape == (num_original_features,)):
        raise ValueError(f"Mean/std stats must be numpy arrays of shape ({num_original_features},)")

    x_norm = (x_original - mean_stats) / std_stats
    x_norm_imputed_zeros = np.nan_to_num(x_norm, nan=0.0)

    x_last_obsv_norm_original = np.zeros_like(x_norm_imputed_zeros, dtype=np.float32)
    current_last_val_norm = np.zeros(num_original_features, dtype=np.float32)
    for t in range(seq_len):
        observed_mask_t = m_original[t] == 1
        current_last_val_norm[observed_mask_t] = x_norm_imputed_zeros[t, observed_mask_t]
        x_last_obsv_norm_original[t, :] = current_last_val_norm

    delta_original = np.zeros_like(x_norm_imputed_zeros, dtype=np.float32)
    for t in range(seq_len):
        if t == 0:
            delta_original[t, :] = 0.0 
        else:
            delta_original[t, :] = delta_original[t-1, :] + 1.0 
        observed_mask_t = m_original[t] == 1
        delta_original[t, observed_mask_t] = 0.0

    x_model_input = x_norm_imputed_zeros
    m_model_input = m_original
    delta_model_input = delta_original
    x_last_model_input = x_last_obsv_norm_original

    if include_current_sepsis_label:
        sepsis_label_t_feature = y_sepsis_labels_current_t.reshape(-1, 1) 
        x_model_input = np.concatenate([x_norm_imputed_zeros, sepsis_label_t_feature], axis=1)
        m_sepsis_label = np.ones((seq_len, 1), dtype=np.float32) # Sepsis label is always "observed" if included
        m_model_input = np.concatenate([m_original, m_sepsis_label], axis=1)
        delta_sepsis_label = np.zeros((seq_len, 1), dtype=np.float32) # Delta for sepsis label is 0
        delta_model_input = np.concatenate([delta_original, delta_sepsis_label], axis=1)
        x_last_sepsis_label = sepsis_label_t_feature 
        x_last_model_input = np.concatenate([x_last_obsv_norm_original, x_last_sepsis_label], axis=1)

    if len(y_sepsis_labels_current_t) > 0:
        y_next_t = np.concatenate([y_sepsis_labels_current_t[1:], np.array([y_sepsis_labels_current_t[-1]], dtype=np.float32)])
    else: 
        y_next_t = np.array([], dtype=np.float32)

    return {
        "x": torch.from_numpy(x_model_input),
        "m": torch.from_numpy(m_model_input),
        "delta": torch.from_numpy(delta_model_input),
        "x_last": torch.from_numpy(x_last_model_input),
        "y_current": torch.from_numpy(y_sepsis_labels_current_t), 
        "y_next": torch.from_numpy(y_next_t),                  
        "length": seq_len,
        "patient_df": df # For reference
    }

# ---------------------------
# Main Inference Function
# ---------------------------
def main_inference():
    parser = argparse.ArgumentParser(description="Run GRU-D Sepsis MDN inference and active info seeking simulation.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pt file).")
    parser.add_argument("--patient_file_path", type=str, required=True, help="Path to the single patient data file (CSV/PSV).")
    parser.add_argument("--mean_path", type=str, required=True, help="Path to the mean_stats.npy file.")
    parser.add_argument("--std_path", type=str, required=True, help="Path to the std_stats.npy file.")
    parser.add_argument("--target_prediction_idx", type=int, default=0, 
                        help="Index of the prediction in the output sequence to analyze for active inference (e.g., 0 for y_next[0]). Default: 0.")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Optional maximum sequence length for patient data.")
    # Note: include_current_sepsis_label will be inferred from the checkpoint's args

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # 1. Load Normalization Statistics
    try:
        mean_stats = np.load(args.mean_path)
        std_stats = np.load(args.std_path)
        print(f"Loaded normalization statistics from {args.mean_path} and {args.std_path}")
    except Exception as e:
        print(f"Error loading normalization statistics: {e}. Exiting.")
        return

    # 2. Load Model from Checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist. Exiting.")
        return
    
    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
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
        print(f"Error loading model state_dict: {e}")
        print("This might be due to a mismatch between the saved model architecture and the current definition or loaded args.")
        print(f"Ensure FEATURE_COLUMNS length ({num_original_features}) and include_current_sepsis_label ({model_args.include_current_sepsis_label}) match the trained model.")
        return

    model.eval()
    print(f"Model loaded successfully from {args.checkpoint_path}.")
    print(f"  Model configured with: hidden_size={model_args.hidden_size}, num_gru_layers={model_args.num_gru_layers}, dropout={model_args.dropout}, num_mdn_components={model_args.num_mdn_components}, include_sepsis_label_feature={model_args.include_current_sepsis_label}")


    # 3. Preprocess Single Patient Data
    # Use max_seq_len from command line if provided, otherwise from checkpoint args, finally None
    max_seq_len_to_use = args.max_seq_len if args.max_seq_len is not None else getattr(model_args, 'max_seq_len', None)

    patient_data = preprocess_single_patient_data(
        args.patient_file_path,
        mean_stats,
        std_stats,
        max_seq_len=max_seq_len_to_use,
        include_current_sepsis_label=model_args.include_current_sepsis_label,
        feature_columns_list=FEATURE_COLUMNS,
        label_column_name=LABEL_COLUMN,
        time_column_name=TIME_COLUMN,
        num_original_features=num_original_features
    )

    if not patient_data or patient_data["length"] == 0:
        print(f"Could not process patient data from {args.patient_file_path} or data is empty. Exiting.")
        return
    
    print(f"Successfully processed patient data from {args.patient_file_path}. Sequence length: {patient_data['length']}")

    # 4. Perform Inference and Active Information Seeking Simulation
    if patient_data["length"] <= args.target_prediction_idx :
         print(f"Patient sequence length ({patient_data['length']}) is too short for target_prediction_idx ({args.target_prediction_idx}). Needs to be at least {args.target_prediction_idx + 1} to have an input at this index.")
         print(f"And needs to be at least {args.target_prediction_idx + 2} to make a prediction for y_next[{args.target_prediction_idx}]. Exiting.")
         return
    
    # Print true label for context, if available in data
    if args.target_prediction_idx < len(patient_data["y_next"]):
        true_next_label = patient_data["y_next"][args.target_prediction_idx].item()
        print(f"  Ground truth for y_next[{args.target_prediction_idx}] (if available): {true_next_label:.0f}")
    else:
        print(f"  Ground truth for y_next[{args.target_prediction_idx}] not available (sequence too short or last step)." )


    simulate_active_feature_acquisition(
        model,
        patient_data["x"], patient_data["m"], patient_data["delta"], patient_data["x_last"],
        patient_data["length"],
        args.target_prediction_idx,
        DEVICE,
        FEATURE_COLUMNS, # Pass the global list
        num_original_features,
        model_args.include_current_sepsis_label,
        model_args.num_mdn_components,
        return_results=False
    )

if __name__ == "__main__":
    main_inference() 