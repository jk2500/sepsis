#!/usr/bin/env python
"""
Trains a Partial VAE model for imputing missing features in sepsis patient data,
conditioned on GRU-D hidden states.
"""
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import random

# Attempt to import from local modules
try:
    from .partial_vae import PartialVAE
    from .model_MDN import (
        SepsisGRUDMDN, SepsisDataset, FEATURE_COLUMNS, DEVICE, 
        TIME_COLUMN, LABEL_COLUMN, load_patient_file
    )
except ImportError:
    print("Attempting fallback import for local modules (e.g., when run as script)...")
    from partial_vae import PartialVAE
    from model_MDN import (
        SepsisGRUDMDN, SepsisDataset, FEATURE_COLUMNS, DEVICE, 
        TIME_COLUMN, LABEL_COLUMN, load_patient_file
    )

def apply_artificial_missingness(x_batch, m_batch, missing_rate):
    """
    Introduces artificial missingness to a batch of data.
    Only applies to features that were originally observed.
    Args:
        x_batch (Tensor): Batch of feature data.
        m_batch (Tensor): Batch of original masks.
        missing_rate (float): Probability of an observed feature being artificially masked.
    Returns:
        Tuple[Tensor, Tensor]: x_batch_mod, m_batch_mod with artificial missingness.
    """
    if missing_rate == 0.0:
        return x_batch, m_batch

    m_batch_mod = m_batch.clone()
    x_batch_mod = x_batch.clone()

    # Iterate over each sample and each feature
    for i in range(m_batch.shape[0]): # Iterate over samples in batch
        for j in range(m_batch.shape[1]): # Iterate over features
            if m_batch[i, j] == 1: # If feature is originally observed
                if random.random() < missing_rate:
                    m_batch_mod[i, j] = 0
                    # x_batch_mod[i,j] = 0 # Zero out the feature as well (standard practice for VAEs)
                    # Alternative: leave x_batch_mod[i,j] as is, as the encoder should use the new mask
                    # For consistency with PartialVAE.encode, zeroing it out is better if masked_x_t = x_t * m_t
                    # If encoder takes x_t and m_t separately and then combines, this isn't strictly needed here.
                    # Let's assume the encoder input is x_t_masked = x_t * m_t_mod.
                    # So we don't need to modify x_batch_mod here, only m_batch_mod.
                    # The VAE's encode method does x_t * m_t, so modifying m_t is sufficient.

    return x_batch_mod, m_batch_mod # x_batch_mod is returned for safety, but m_batch_mod is key

def prepare_vae_dataset_from_patient_files(
    patient_files, grud_model, grud_model_args, mean_stats, std_stats, 
    max_seq_len, device, include_current_sepsis_label_in_grud_input):
    """
    Prepares a dataset of (x_t, m_t, h_grud_t) tuples for VAE training.
    """
    grud_model.eval() # Ensure GRU-D is in eval mode
    all_x_t, all_m_t, all_h_grud_t = [], [], []

    num_original_features = len(FEATURE_COLUMNS)

    for p_file in patient_files:
        try:
            # Adapted from preprocess_single_patient_data in inference_active.py
            df = load_patient_file(p_file)
            if max_seq_len is not None:
                df = df.iloc[:max_seq_len]

            x_original_np = df[FEATURE_COLUMNS].values.astype(np.float32)
            seq_len = x_original_np.shape[0]
            if seq_len == 0: continue

            m_original_np = (~np.isnan(x_original_np)).astype(np.float32)
            
            # Normalize as per GRU-D training
            x_norm_np = (x_original_np - mean_stats) / std_stats
            x_norm_imputed_zeros_np = np.nan_to_num(x_norm_np, nan=0.0)

            # Prepare other GRU-D inputs (delta, x_last)
            x_last_obsv_norm_np = np.zeros_like(x_norm_imputed_zeros_np, dtype=np.float32)
            current_last_val_norm = np.zeros(num_original_features, dtype=np.float32)
            for t_idx in range(seq_len):
                observed_mask_t = m_original_np[t_idx] == 1
                current_last_val_norm[observed_mask_t] = x_norm_imputed_zeros_np[t_idx, observed_mask_t]
                x_last_obsv_norm_np[t_idx, :] = current_last_val_norm

            delta_np = np.zeros_like(x_norm_imputed_zeros_np, dtype=np.float32)
            for t_idx in range(seq_len):
                if t_idx == 0: delta_np[t_idx, :] = 0.0 
                else: delta_np[t_idx, :] = delta_np[t_idx-1, :] + 1.0 
                observed_mask_t = m_original_np[t_idx] == 1
                delta_np[t_idx, observed_mask_t] = 0.0

            # Concatenate sepsis label if GRU-D expects it
            x_grud_input_np = x_norm_imputed_zeros_np
            m_grud_input_np = m_original_np
            delta_grud_input_np = delta_np
            x_last_grud_input_np = x_last_obsv_norm_np

            if include_current_sepsis_label_in_grud_input:
                y_sepsis_labels_current_t_np = df[LABEL_COLUMN].values.astype(np.float32).reshape(-1,1)
                x_grud_input_np = np.concatenate([x_norm_imputed_zeros_np, y_sepsis_labels_current_t_np], axis=1)
                m_sepsis_label_np = np.ones((seq_len, 1), dtype=np.float32)
                m_grud_input_np = np.concatenate([m_original_np, m_sepsis_label_np], axis=1)
                delta_sepsis_label_np = np.zeros((seq_len, 1), dtype=np.float32)
                delta_grud_input_np = np.concatenate([delta_np, delta_sepsis_label_np], axis=1)
                x_last_sepsis_label_np = y_sepsis_labels_current_t_np
                x_last_grud_input_np = np.concatenate([x_last_obsv_norm_np, x_last_sepsis_label_np], axis=1)
            
            # Convert to tensors for GRU-D
            x_grud_batch = torch.from_numpy(x_grud_input_np).unsqueeze(0).to(device)
            m_grud_batch = torch.from_numpy(m_grud_input_np).unsqueeze(0).to(device)
            delta_grud_batch = torch.from_numpy(delta_grud_input_np).unsqueeze(0).to(device)
            x_last_grud_batch = torch.from_numpy(x_last_grud_input_np).unsqueeze(0).to(device)
            lengths_batch = torch.tensor([seq_len], device=device, dtype=torch.long)

            with torch.no_grad():
                # We need raw GRU output (hidden states from all layers or last layer)
                # Modify SepsisGRUDMDN to return hidden states if it doesn't already
                # For now, assume `grud_model.get_hidden_states(...)` exists or adapt its forward
                # The `gru_d` in SepsisGRUDMDN outputs (output, hidden), where output is from all timesteps.
                # SepsisGRUDMDN.forward calls self.gru_d(x, m, delta, x_last_obsv_norm, lengths)
                # Let's get the direct output of self.gru_d
                grud_output_raw, _ = grud_model.gru_d(x_grud_batch, m_grud_batch, delta_grud_batch, x_last_grud_batch, lengths_batch)
                # grud_output_raw shape: (batch_size, seq_len, hidden_size)
            
            h_grud_sequence = grud_output_raw.squeeze(0) # (seq_len, hidden_size)

            # VAE uses original features (normalized, imputed) and original masks
            x_vae_input_t = torch.from_numpy(x_norm_imputed_zeros_np).to(device)
            m_vae_input_t = torch.from_numpy(m_original_np).to(device)

            for t in range(seq_len):
                all_x_t.append(x_vae_input_t[t])
                all_m_t.append(m_vae_input_t[t])
                all_h_grud_t.append(h_grud_sequence[t])
        except Exception as e:
            print(f"Error processing file {p_file}: {e}. Skipping.")
            continue
    
    if not all_x_t:
        raise ValueError("No data processed for VAE training. Check input files and paths.")

    return TensorDataset(torch.stack(all_x_t), torch.stack(all_m_t), torch.stack(all_h_grud_t))

def main():
    parser = argparse.ArgumentParser(description="Train Partial VAE for Sepsis Feature Imputation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing patient (PSV/CSV) files for training.")
    parser.add_argument("--grud_checkpoint_path", type=str, required=True, help="Path to pre-trained GRU-D model checkpoint (.pt).")
    parser.add_argument("--vae_output_dir", type=str, required=True, help="Directory to save trained Partial VAE model.")
    parser.add_argument("--mean_path", type=str, required=True, help="Path to mean_stats.npy for normalization.")
    parser.add_argument("--std_path", type=str, required=True, help="Path to std_stats.npy for normalization.")
    
    parser.add_argument("--max_seq_len", type=int, default=None, help="Optional max sequence length for patient data.")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension for VAE.")
    parser.add_argument("--vae_encoder_hidden_dims", type=int, nargs='+', default=[128, 64], help="Encoder hidden layer dimensions.")
    parser.add_argument("--vae_decoder_hidden_dims", type=int, nargs='+', default=[64, 128], help="Decoder hidden layer dimensions.")
    parser.add_argument("--artificial_missing_rate", type=float, default=0.1, help="Rate of artificial missingness for VAE training (0 to disable).")
    parser.add_argument("--beta_vae", type=float, default=1.0, help="Beta hyperparameter for VAE loss (KL divergence weight).")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs for VAE.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for VAE training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for VAE optimizer.")
    parser.add_argument("--num_workers_loader", type=int, default=0, help="Num workers for DataLoader.")

    args = parser.parse_args()
    os.makedirs(args.vae_output_dir, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # 1. Load Normalization Statistics
    try:
        mean_stats = np.load(args.mean_path)
        std_stats = np.load(args.std_path)
    except Exception as e:
        print(f"Error loading normalization statistics: {e}. Exiting.")
        return

    # 2. Load pre-trained GRU-D model
    if not os.path.exists(args.grud_checkpoint_path):
        print(f"Error: GRU-D checkpoint {args.grud_checkpoint_path} not found. Exiting.")
        return
    
    grud_checkpoint = torch.load(args.grud_checkpoint_path, map_location=DEVICE)
    grud_model_args = grud_checkpoint.get('args')
    if not grud_model_args:
        print("Error: GRU-D checkpoint does not contain 'args'. Exiting.")
        return

    num_original_features = len(FEATURE_COLUMNS)
    num_grud_input_features = num_original_features + (1 if grud_model_args.include_current_sepsis_label else 0)

    grud_model = SepsisGRUDMDN(
        input_size=num_grud_input_features,
        hidden_size=grud_model_args.hidden_size,
        num_gru_layers=grud_model_args.num_gru_layers,
        dropout=0, # No dropout for feature extraction
        num_mdn_components=grud_model_args.num_mdn_components
    ).to(DEVICE)
    grud_model.load_state_dict(grud_checkpoint['model_state_dict'])
    grud_model.eval()
    print("Pre-trained GRU-D model loaded successfully.")

    # 3. Prepare VAE training dataset
    print("Preparing dataset for VAE training...")
    patient_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(('.psv', '.csv'))]
    if not patient_files:
        print(f"No patient files found in {args.data_dir}. Exiting.")
        return
    
    vae_dataset = prepare_vae_dataset_from_patient_files(
        patient_files, grud_model, grud_model_args, mean_stats, std_stats, 
        args.max_seq_len, DEVICE, grud_model_args.include_current_sepsis_label
    )
    vae_dataloader = DataLoader(vae_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_loader)
    print(f"VAE dataset prepared with {len(vae_dataset)} total timesteps.")

    # 4. Initialize PartialVAE model and optimizer
    # VAE feature_dim is based on original features (no sepsis label)
    vae_feature_dim = num_original_features 
    vae_hidden_dim_grud = grud_model_args.hidden_size

    partial_vae_model = PartialVAE(
        feature_dim=vae_feature_dim,
        hidden_dim_grud=vae_hidden_dim_grud,
        latent_dim=args.latent_dim,
        encoder_hidden_dims=args.vae_encoder_hidden_dims,
        decoder_hidden_dims=args.vae_decoder_hidden_dims
    ).to(DEVICE)
    
    optimizer = optim.Adam(partial_vae_model.parameters(), lr=args.lr)
    print("PartialVAE model and optimizer initialized.")

    # 5. Training loop
    print("Starting PartialVAE training...")
    partial_vae_model.train()
    for epoch in range(args.epochs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        num_batches = 0

        for x_t_batch, m_t_batch, h_grud_t_batch in vae_dataloader:
            x_t_batch = x_t_batch.to(DEVICE)
            m_t_batch = m_t_batch.to(DEVICE)
            h_grud_t_batch = h_grud_t_batch.to(DEVICE)

            # Apply artificial missingness for VAE training
            # Note: x_t_batch here is already normalized and imputed with zeros for true missing values.
            # Artificial missingness will further mask some of the *observed* values in m_t_batch.
            _, m_t_batch_mod = apply_artificial_missingness(x_t_batch, m_t_batch, args.artificial_missing_rate)
            
            optimizer.zero_grad()
            x_prime_mu, z_mean, z_logvar = partial_vae_model(x_t_batch, m_t_batch_mod, h_grud_t_batch)
            
            total_loss, recon_loss, kld = partial_vae_model.loss_function(
                x_t_batch, m_t_batch_mod, # Loss is w.r.t. the artificially masked data
                x_prime_mu, z_mean, z_logvar, beta=args.beta_vae
            )
            
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld.item()
            num_batches += 1
        
        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kld_loss = epoch_kld_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_total_loss:.4f}, Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f}")

    # 6. Save the trained VAE model
    vae_checkpoint_path = os.path.join(args.vae_output_dir, "partial_vae_checkpoint.pt")
    torch.save({
        'model_state_dict': partial_vae_model.state_dict(),
        'args': args, # Save training args for reference
        'feature_dim': vae_feature_dim,
        'hidden_dim_grud': vae_hidden_dim_grud,
        'latent_dim': args.latent_dim,
        'encoder_hidden_dims': args.vae_encoder_hidden_dims,
        'decoder_hidden_dims': args.vae_decoder_hidden_dims,
        'grud_model_args': grud_model_args # Save GRU-D args for context
    }, vae_checkpoint_path)
    print(f"Trained PartialVAE model saved to {vae_checkpoint_path}")

if __name__ == "__main__":
    main() 