#!/usr/bin/env python
"""
Train a GRU-D model with an MDN head to predict the N-hours-ahead sepsis label 
and its uncertainty from PhysioNet Sepsis Challenge data, with weighted loss for transitions.

Usage (example):
    python grud_sepsis_mdn_weighted.py --data_dir /path/to/training_data --epochs 30 --batch_size 64 --lr 1e-3 \
                                      --num_mdn_components 3 --transition_weight 5.0 --prediction_horizon 6 \
                                      --include_current_sepsis_label --cache_data

Expecting each patient record in CSV/PSV format with a column named "SepsisLabel" and the standard feature set
from the 2019 PhysioNet/Computing in Cardiology Challenge.
"""
import argparse
import os
from glob import glob
import time # For timing epochs
import hashlib # For caching

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split # For train/val split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F # For F.softplus, F.log_softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import scheduler
from torch.utils.tensorboard import SummaryWriter # Import TensorBoard

# ---------------------------
# Hyper‑parameters (defaults)
# ---------------------------
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 30
DEFAULT_MAX_SEQ_LEN = None  # use full length
DEFAULT_NUM_GRU_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_NUM_MDN_COMPONENTS = 3
DEFAULT_TRANSITION_WEIGHT = 1.0 # For weighted loss
DEFAULT_PREDICTION_HORIZON = 1 # Predict 1 hour in advance by default

# ---------------------------
# Device selection — now MPS‑aware
# ---------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ---------------------------
# Feature list (2019 Challenge)
# ---------------------------
FEATURE_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3",
    "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
    "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "Gender", "HospAdmTime", "ICULOS",
]
LABEL_COLUMN = "SepsisLabel"
TIME_COLUMN = "Time"

# ---------------------------
# Utility functions
# ---------------------------
def positive_int(value):
    """Custom argparse type for positive integers (>=1)."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value for prediction_horizon (must be >= 1)")
    return ivalue

def load_patient_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|" if path.endswith(".psv") else ",")
    df[TIME_COLUMN] = np.arange(len(df))
    return df

def compute_normalisation_stats(file_paths):
    sums = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    sq_sums = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    count = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    for fp in file_paths:
        data = load_patient_file(fp)[FEATURE_COLUMNS].values.astype(np.float32)
        mask = ~np.isnan(data)
        sums += np.nansum(data, axis=0)
        sq_sums += np.nansum(data**2, axis=0) 
        count += mask.sum(axis=0)
    mean = sums / np.maximum(count, 1)
    std = np.sqrt(np.maximum(0, sq_sums / np.maximum(count, 1) - mean ** 2)) 
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

# ---------------------------
# Dataset & DataLoader
# ---------------------------
class SepsisDataset(Dataset):
    def __init__(self, file_paths, mean, std, prediction_horizon: int, 
                 max_seq_len: int | None = None, 
                 include_current_sepsis_label: bool = False, 
                 cache_dir: str | None = None, dataset_type: str = "train"): 
        self.mean = mean 
        self.std = std   
        self.num_original_features = len(FEATURE_COLUMNS)
        self.prediction_horizon = prediction_horizon # Store prediction horizon
        self.max_seq_len = max_seq_len
        self.include_current_sepsis_label = include_current_sepsis_label
        self.items = []

        cache_attempted = False
        if cache_dir:
            cache_attempted = True
            os.makedirs(cache_dir, exist_ok=True)
            
            # Include prediction_horizon in cache key
            relevant_args_str = (f"max_seq_len={self.max_seq_len}_"
                                 f"include_label={self.include_current_sepsis_label}_"
                                 f"horizon={self.prediction_horizon}")
            files_hash = hashlib.md5(str(sorted(file_paths)).encode()).hexdigest()
            mean_std_hash = hashlib.md5(self.mean.tobytes() + self.std.tobytes()).hexdigest()
            
            cache_filename = f"{dataset_type}_cache_{files_hash}_{relevant_args_str}_{mean_std_hash}.pt"
            self.cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(self.cache_path):
                print(f"Attempting to load {dataset_type} data from cache: {self.cache_path}")
                try:
                    cached_content = torch.load(self.cache_path, map_location='cpu') 
                    
                    cached_meta = cached_content.get('metadata', {})
                    expected_num_files = len(file_paths)
                    cached_num_files = cached_meta.get('num_files')
                    
                    # Check all relevant args from snapshot
                    cached_args_snapshot = cached_meta.get('args_snapshot', {})
                    cached_max_seq_len = cached_args_snapshot.get('max_seq_len')
                    cached_include_label = cached_args_snapshot.get('include_current_sepsis_label')
                    cached_prediction_horizon = cached_args_snapshot.get('prediction_horizon')


                    if (cached_num_files == expected_num_files and
                        cached_max_seq_len == self.max_seq_len and
                        cached_include_label == self.include_current_sepsis_label and
                        cached_prediction_horizon == self.prediction_horizon): # Check horizon
                        self.items = cached_content['data']
                        print(f"Successfully loaded {len(self.items)} items for {dataset_type} from cache.")
                    else:
                        print("Cache metadata mismatch (num_files or critical args including horizon). Re-processing.")
                        self.items = [] # Ensure re-processing
                except Exception as e:
                    print(f"Error loading from cache: {e}. Re-processing.")
                    self.items = [] # Ensure re-processing
        
        if not self.items: 
            if cache_attempted and hasattr(self, 'cache_path'): # Check if cache was attempted and path exists
                 print(f"Processing {dataset_type} data and saving to cache: {self.cache_path}")
            elif cache_attempted and not hasattr(self, 'cache_path'): # Should not happen if cache_dir provided
                 print(f"Processing {dataset_type} data (cache path not set despite cache_dir).")
            else: # cache_dir was None
                 print(f"Processing {dataset_type} data (caching disabled).")
            
            self.items = self._process_all_files(file_paths)
            
            if cache_attempted and hasattr(self, 'cache_path') and self.items: 
                metadata_to_save = {
                    'num_files': len(file_paths),
                    'args_snapshot': { 
                        'max_seq_len': self.max_seq_len,
                        'include_current_sepsis_label': self.include_current_sepsis_label,
                        'prediction_horizon': self.prediction_horizon # Save horizon
                    }
                }
                try:
                    torch.save({'data': self.items, 'metadata': metadata_to_save}, self.cache_path)
                    print(f"Saved {len(self.items)} processed items for {dataset_type} to {self.cache_path}")
                except Exception as e:
                    print(f"Error saving cache to {self.cache_path}: {e}")

    def _process_all_files(self, file_paths):
        processed_items = []
        total_files = len(file_paths)
        print_interval = max(1, total_files // 20) 

        for i, file_path in enumerate(file_paths):
            if (i + 1) % print_interval == 0 or i == total_files - 1:
                 print(f"  Processing file {i+1}/{total_files} for dataset...")
            item_data = self._get_single_item_data_from_file(file_path)
            if item_data["length"] > 0 : # Only add non-empty sequences
                processed_items.append(item_data)
        return processed_items

    def _get_single_item_data_from_file(self, file_path): 
        df = load_patient_file(file_path)
        if self.max_seq_len is not None:
            df = df.iloc[: self.max_seq_len]
            
        x_original = df[FEATURE_COLUMNS].values.astype(np.float32)
        y_sepsis_labels_at_t = df[LABEL_COLUMN].values.astype(np.float32) # SepsisLabel_t

        seq_len = x_original.shape[0]
        # Handle empty or too short sequences early to avoid errors
        num_input_features = self.num_original_features + (1 if self.include_current_sepsis_label else 0)
        if seq_len == 0: 
            return {
                "x": torch.empty((0, num_input_features), dtype=torch.float32),
                "m": torch.empty((0, num_input_features), dtype=torch.float32),
                "delta": torch.empty((0, num_input_features), dtype=torch.float32),
                "x_last": torch.empty((0, num_input_features), dtype=torch.float32),
                "y_current": torch.empty(0, dtype=torch.float32),
                "y_target": torch.empty(0, dtype=torch.float32), # Changed from y_next
                "length": 0,
            }

        m_original = (~np.isnan(x_original)).astype(np.float32)
        x_norm = (x_original - self.mean) / self.std 
        x_norm_imputed_zeros = np.nan_to_num(x_norm, nan=0.0)

        x_last_obsv_norm_original = np.zeros_like(x_norm_imputed_zeros, dtype=np.float32)
        current_last_val_norm = np.zeros(self.num_original_features, dtype=np.float32)
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

        if self.include_current_sepsis_label:
            sepsis_label_t_feature = y_sepsis_labels_at_t.reshape(-1, 1) 
            x_model_input = np.concatenate([x_norm_imputed_zeros, sepsis_label_t_feature], axis=1)
            m_sepsis_label = np.ones((seq_len, 1), dtype=np.float32)
            m_model_input = np.concatenate([m_original, m_sepsis_label], axis=1)
            delta_sepsis_label = np.zeros((seq_len, 1), dtype=np.float32)
            delta_model_input = np.concatenate([delta_original, delta_sepsis_label], axis=1)
            x_last_sepsis_label = sepsis_label_t_feature 
            x_last_model_input = np.concatenate([x_last_obsv_norm_original, x_last_sepsis_label], axis=1)

        # Target label: SepsisLabel_{t + prediction_horizon}
        y_target_sequence = np.full(seq_len, np.nan, dtype=np.float32) # Initialize with NaN
        if len(y_sepsis_labels_at_t) > 0 : # Redundant given seq_len check, but safe
            # Pad with the last known label by default for timesteps beyond valid prediction range
            y_target_sequence.fill(y_sepsis_labels_at_t[-1]) 
            
            num_valid_target_predictions = seq_len - self.prediction_horizon
            
            if num_valid_target_predictions > 0:
                valid_targets = y_sepsis_labels_at_t[self.prediction_horizon : self.prediction_horizon + num_valid_target_predictions]
                y_target_sequence[:num_valid_target_predictions] = valid_targets
        else: # This case should be caught by seq_len == 0 check earlier
            y_target_sequence = np.array([], dtype=np.float32)


        return {
            "x": torch.from_numpy(x_model_input),
            "m": torch.from_numpy(m_model_input),
            "delta": torch.from_numpy(delta_model_input),
            "x_last": torch.from_numpy(x_last_model_input),
            "y_current": torch.from_numpy(y_sepsis_labels_at_t), 
            "y_target": torch.from_numpy(y_target_sequence), # Changed from y_next                  
            "length": seq_len, 
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    # Filter out items with length 0 that might have slipped through (e.g., if _process_all_files didn't filter)
    batch = [b for b in batch if b["length"] > 0]
    if not batch: # If all items were filtered or batch was initially empty
        num_x_features = len(FEATURE_COLUMNS) # Fallback, ideally get from args or a non-empty item
        # Consider if include_current_sepsis_label is true, num_x_features would be +1
        # This edge case handling for totally empty batches needs care if used.
        return (torch.empty((0, 0, num_x_features)), torch.empty((0, 0, num_x_features)), 
                torch.empty((0, 0, num_x_features)), torch.empty((0, 0, num_x_features)), 
                torch.empty((0, 0)), torch.empty((0,0)), torch.tensor([], dtype=torch.long))

    batch.sort(key=lambda b: b["length"], reverse=True)
    lengths = [b["length"] for b in batch]
    max_len = lengths[0] # batch is guaranteed not empty here

    num_x_features = batch[0]["x"].size(1) # Get from the first item

    def pad_tensor(tensors_list, is_label_type):
        # Pad with 0.0. Loss mask handles validity.
        return torch.stack([torch.nn.functional.pad(t, (0, max_len - len(t)), value=0.0) for t in tensors_list])

    def pad_feature_tensor(tensors_list):
        return torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_len - t.size(0))) for t in tensors_list])

    x = pad_feature_tensor([b["x"] for b in batch])
    m = pad_feature_tensor([b["m"] for b in batch])
    delta = pad_feature_tensor([b["delta"] for b in batch])
    x_last = pad_feature_tensor([b["x_last"] for b in batch])
    y_current = pad_tensor([b["y_current"] for b in batch], True)
    y_target = pad_tensor([b["y_target"] for b in batch], True) # Changed from y_next
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return x, m, delta, x_last, y_current, y_target, lengths_tensor

# ---------------------------
# GRU‑D Model Components
# (No changes needed in GRUDCell, GRUD for arbitrary horizon)
# ---------------------------
class GRUDCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma_x_decay = nn.Parameter(torch.Tensor(input_size))
        self.gamma_h_decay = nn.Parameter(torch.Tensor(hidden_size)) # For feature-wise hidden decay
        self.W_r = nn.Linear(input_size * 3, hidden_size)
        self.W_z = nn.Linear(input_size * 3, hidden_size)
        self.W_h_tilde = nn.Linear(input_size * 3, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_h_tilde = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.W_r, self.W_z, self.W_h_tilde, self.U_r, self.U_z, self.U_h_tilde]:
            if hasattr(layer, 'weight'): nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None: nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gamma_x_decay)
        nn.init.zeros_(self.gamma_h_decay) 

    def forward(self, x_t, m_t, delta_t, x_last_obsv_norm, h_prev):
        gamma_x_t = torch.exp(-torch.relu(self.gamma_x_decay) * delta_t) # (batch, input_size)
        x_hat_t = m_t * x_t + (1 - m_t) * (gamma_x_t * x_last_obsv_norm) # (batch, input_size)
        
        # Hidden state decay factor for GRU-D (Che et al. 2018) is element-wise based on delta_t
        # The original paper suggests gamma_h_decay is (hidden_size), and delta_t is implicitly 1 for h.
        # If using feature-wise delta for hidden state (more complex):
        # delta_t_mean = torch.mean(delta_t, dim=1, keepdim=True) # (batch, 1)
        # gamma_h_t_factor = torch.exp(-torch.relu(self.gamma_h_decay) * delta_t_mean) # (batch, hidden_size)

        # Standard GRU-D hidden decay (assumes fixed inter-step time for h)
        gamma_h_t_factor = torch.exp(-torch.relu(self.gamma_h_decay)) # (hidden_size), broadcasts with h_prev
        
        h_prev_decayed = gamma_h_t_factor * h_prev # (batch, hidden_size)
        
        concat_for_gates = torch.cat([x_hat_t, m_t, gamma_x_t], dim=1) # (batch, input_size * 3)
        r_t = torch.sigmoid(self.W_r(concat_for_gates) + self.U_r(h_prev_decayed))
        z_t = torch.sigmoid(self.W_z(concat_for_gates) + self.U_z(h_prev_decayed))
        h_tilde_t = torch.tanh(self.W_h_tilde(concat_for_gates) + self.U_h_tilde(r_t * h_prev_decayed))
        h_curr = (1 - z_t) * h_prev_decayed + z_t * h_tilde_t
        return h_curr, x_hat_t

class GRUD(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size; self.hidden_size = hidden_size; self.num_layers = num_layers
        self.cells = nn.ModuleList()
        current_dim = input_size
        for i in range(num_layers):
            self.cells.append(GRUDCell(current_dim, hidden_size))
            current_dim = hidden_size 
        self.dropout_layer = None
        if dropout > 0.0 and num_layers > 1: self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, m, delta, x_last, lengths=None):
        batch_size, seq_len, _ = x.size()
        h_layer_states = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        outputs_from_last_layer_seq = []
        for t in range(seq_len):
            x_t_layer_input = x[:, t, :]; m_t_layer_input = m[:, t, :]; delta_t_layer_input = delta[:, t, :]; x_last_t_layer_input = x_last[:, t, :]
            for l_idx in range(self.num_layers):
                h_prev_for_cell = h_layer_states[l_idx]
                # For layers > 0, input is hidden state from previous layer. Mask is all ones, delta is all zeros.
                if l_idx > 0: 
                    m_t_layer_input_cell = torch.ones_like(x_t_layer_input)
                    delta_t_layer_input_cell = torch.zeros_like(x_t_layer_input)
                    x_last_t_layer_input_cell = x_t_layer_input # Current value is "last observed"
                else: # First layer uses the actual inputs
                    m_t_layer_input_cell = m_t_layer_input
                    delta_t_layer_input_cell = delta_t_layer_input
                    x_last_t_layer_input_cell = x_last_t_layer_input

                h_new_for_cell, _ = self.cells[l_idx](x_t_layer_input, m_t_layer_input_cell, 
                                                    delta_t_layer_input_cell, x_last_t_layer_input_cell, 
                                                    h_prev_for_cell)
                h_layer_states[l_idx] = h_new_for_cell
                x_t_layer_input = h_new_for_cell # Output of this layer is input x for the next
                if self.dropout_layer and l_idx < self.num_layers - 1: 
                    x_t_layer_input = self.dropout_layer(x_t_layer_input)
            outputs_from_last_layer_seq.append(h_layer_states[-1])
        outputs_stacked = torch.stack(outputs_from_last_layer_seq, dim=1)
        return outputs_stacked, h_layer_states

# ---------------------------
# Main Sepsis Prediction Model with MDN
# ---------------------------
class SepsisGRUDMDN(nn.Module):
    def __init__(self, input_size, hidden_size, num_gru_layers=1, dropout=0.0, num_mdn_components=3):
        super().__init__()
        self.grud = GRUD(input_size, hidden_size, num_gru_layers, dropout)
        self.num_mdn_components = num_mdn_components
        # MDN for Bernoulli: K mixture weights (pi_logits), K means (mu) for logits
        # Sigmas are not directly part of Bernoulli params but can model uncertainty in logits
        self.fc_mdn = nn.Linear(hidden_size, num_mdn_components * 2) # pi_logits, mus

    def forward(self, x, m, delta, x_last, lengths=None):
        grud_output_seq, _ = self.grud(x, m, delta, x_last, lengths)
        mdn_params = self.fc_mdn(grud_output_seq)
        # Split into pi_logits, mus
        pi_logits, mus = torch.split(mdn_params, self.num_mdn_components, dim=-1)
        pis = F.softmax(pi_logits, dim=-1)
        # Sigmas are not output by this MDN version, but can be fixed or learned differently if needed
        return pis, mus

# ---------------------------
# MDN Loss Function
# ---------------------------
def mdn_loss_bernoulli(pis, mus, targets, sample_weights=None): # targets is y_target
    targets_expanded = targets.unsqueeze(-1) 
    log_prob_y_given_component_k = -(F.binary_cross_entropy_with_logits(
        mus, targets_expanded.expand_as(mus), reduction='none'
    ))
    log_pis = torch.log(pis + 1e-8)
    weighted_log_probs = log_pis + log_prob_y_given_component_k
    log_likelihood = torch.logsumexp(weighted_log_probs, dim=-1) # (batch, seq_len)
    
    nll_per_timestep = -log_likelihood # (batch, seq_len)

    if sample_weights is not None:
        nll_per_timestep = nll_per_timestep * sample_weights

    return nll_per_timestep

# ---------------------------
# Training and Evaluation
# ---------------------------
def train_epoch(model, dataloader, optimizer, device, num_mdn_components, 
                transition_weight: float, prediction_horizon: int, scaler=None): # Added prediction_horizon
    model.train()
    total_loss_sum = 0.0
    num_valid_timesteps_for_loss = 0

    for x, m, delta, x_last, y_current, y_target, lengths in dataloader: # y_target instead of y_next
        x, m, delta, x_last, y_current, y_target, lengths = (
            x.to(device), m.to(device), delta.to(device), 
            x_last.to(device), y_current.to(device), y_target.to(device), lengths.to(device)
        )
        
        optimizer.zero_grad(set_to_none=True)

        autocast_enabled = scaler is not None and device.type == 'cuda'

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            pis, mus = model(x, m, delta, x_last, lengths)
            
            # Loss mask based on prediction_horizon
            loss_mask = torch.zeros_like(y_target, dtype=torch.bool, device=device)
            for i, l_val in enumerate(lengths):
                if l_val > prediction_horizon:
                    num_valid_preds_for_seq = l_val - prediction_horizon
                    loss_mask[i, :num_valid_preds_for_seq] = True

            if loss_mask.sum() == 0:
                continue

            # Compute main sample weights (1 by default)
            sample_weights = torch.ones_like(y_target, device=device)

            # Compute full loss (weighted) — this is the baseline loss
            nll_per_timestep_weighted = mdn_loss_bernoulli(pis, mus, y_target, sample_weights)

            # Now compute transition-only loss
            is_0_to_1_transition = (y_current == 0) & (y_target == 1) & loss_mask

            if is_0_to_1_transition.sum() > 0:
                transition_loss = nll_per_timestep_weighted[is_0_to_1_transition].mean()
                loss = nll_per_timestep_weighted[loss_mask].mean() + transition_weight * transition_loss
            else:
                loss = nll_per_timestep_weighted[loss_mask].mean()

            

        if scaler and device.type == 'cuda':
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else: # Standard precision or non-CUDA mixed precision (not typical)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss_sum += nll_per_timestep_weighted[loss_mask].sum().item()
        num_valid_timesteps_for_loss += loss_mask.sum().item()

    return total_loss_sum / num_valid_timesteps_for_loss if num_valid_timesteps_for_loss > 0 else 0.0


def evaluate(model, dataloader, device, prediction_horizon: int): # Added prediction_horizon
    model.eval()
    total_loss_sum_eval = 0.0
    num_valid_timesteps_for_loss_eval = 0
    all_true_target_labels_list = [] # SepsisLabel_{t+N}
    all_pred_target_probs_list = []  # P(SepsisLabel_{t+N}=1 | X_t)
    all_logit_variances_list = []
    all_current_labels_list = []    # SepsisLabel_t

    autocast_enabled_eval = device.type == 'cuda' # Enable for eval if CUDA for consistency, though less critical

    with torch.no_grad():
        for x, m, delta, x_last, y_current, y_target, lengths in dataloader: # y_target
            x, m, delta, x_last, y_current, y_target, lengths = (
                x.to(device), m.to(device), delta.to(device), 
                x_last.to(device), y_current.to(device), y_target.to(device), lengths.to(device)
            )

            with torch.cuda.amp.autocast(enabled=autocast_enabled_eval):
                pis, mus = model(x, m, delta, x_last, lengths)
            
            # Loss mask based on prediction_horizon
            loss_mask = torch.zeros_like(y_target, dtype=torch.bool, device=device)
            for i, l_val in enumerate(lengths):
                 if l_val > prediction_horizon:
                    num_valid_preds_for_seq = l_val - prediction_horizon
                    loss_mask[i, :num_valid_preds_for_seq] = True
            
            if loss_mask.sum() == 0:
                continue
            
            # Unweighted NLL for evaluation, using y_target
            nll_per_timestep_eval = mdn_loss_bernoulli(pis, mus, y_target, sample_weights=None)
            # loss_eval = nll_per_timestep_eval[loss_mask].mean() # Not needed if just summing
            total_loss_sum_eval += nll_per_timestep_eval[loss_mask].sum().item()
            num_valid_timesteps_for_loss_eval += loss_mask.sum().item()

            probs_k = torch.sigmoid(mus)
            mean_probs = torch.sum(pis * probs_k, dim=-1) # P(SepsisLabel_{t+N}=1 | X_t)
            
            # Logit variance (assuming sigma is implicitly 0 for Bernoulli from logit)
            expected_logit = torch.sum(pis * mus, dim=-1)
            # Var(logit) = E[logit^2] - (E[logit])^2. E[logit^2] = sum pi_k * mu_k^2 (since sigma_k=0 for this MDN version)
            expected_logit_sq = torch.sum(pis * (mus.pow(2)), dim=-1) 
            logit_variance = torch.relu(expected_logit_sq - expected_logit.pow(2))

            all_true_target_labels_list.extend(y_target[loss_mask].cpu().numpy())
            all_pred_target_probs_list.extend(mean_probs[loss_mask].cpu().numpy())
            all_logit_variances_list.extend(logit_variance[loss_mask].cpu().numpy())
            all_current_labels_list.extend(y_current[loss_mask].cpu().numpy())
            
    avg_loss_eval = total_loss_sum_eval / num_valid_timesteps_for_loss_eval if num_valid_timesteps_for_loss_eval > 0 else 0.0
    
    if not all_true_target_labels_list: # Renamed list
        return avg_loss_eval, 0.0, 0.0, np.array([]), np.nan

    np_true_target_labels = np.array(all_true_target_labels_list) # Renamed
    np_pred_target_probs = np.array(all_pred_target_probs_list)   # Renamed

    try:
        auc = roc_auc_score(np_true_target_labels, np_pred_target_probs)
    except ValueError:
        auc = 0.0
        
    accuracy = accuracy_score(np_true_target_labels, (np_pred_target_probs > 0.5).astype(int))
    
    transition_recall_0_to_1 = np.nan
    if all_current_labels_list: # Check if list has content
        np_current_labels = np.array(all_current_labels_list)
        # Transition from S_t=0 to S_{t+N}=1
        actual_0_to_1_mask = (np_current_labels == 0) & (np_true_target_labels == 1)
        num_actual_0_to_1_transitions = np.sum(actual_0_to_1_mask)
        if num_actual_0_to_1_transitions > 0:
            preds_for_0_to_1_transitions = np_pred_target_probs[actual_0_to_1_mask]
            num_correctly_predicted_0_to_1 = np.sum(preds_for_0_to_1_transitions > 0.5)
            transition_recall_0_to_1 = num_correctly_predicted_0_to_1 / num_actual_0_to_1_transitions
            
    return avg_loss_eval, auc, accuracy, np.array(all_logit_variances_list), transition_recall_0_to_1


# ---------------------------
# Main script
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train GRU-D with MDN for Sepsis Prediction with weighted loss and N-hr horizon")
    parser.add_argument("--data_dir", type=str, default="physionet.org/", help="Directory containing patient CSV/PSV files")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE, help="GRU-D hidden size")
    parser.add_argument("--num_gru_layers", type=int, default=DEFAULT_NUM_GRU_LAYERS, help="Number of GRU-D layers")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate between GRU-D layers")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data for validation (0 to disable)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--prediction_horizon", type=positive_int, default=DEFAULT_PREDICTION_HORIZON, # New argument
                        help="Number of hours in advance to predict (horizon, must be >= 1). Default: %(default)s hours.")
    parser.add_argument("--include_current_sepsis_label", action="store_true", 
                        help="Include SepsisLabel_t as an input feature to the model (for MDN, this is separate from y_current for loss).")
    parser.add_argument("--num_mdn_components", type=int, default=DEFAULT_NUM_MDN_COMPONENTS, help="Number of MDN mixture components")
    parser.add_argument("--transition_weight", type=float, default=DEFAULT_TRANSITION_WEIGHT,
                        help="Weight for S_t=0 -> S_{t+N}=1 sepsis transitions in the loss function. Default 1.0.")
    parser.add_argument("--torch_compile", action="store_true", 
                        help="Enable torch.compile() for the model (requires PyTorch 2.0+).")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training (torch.cuda.amp) if CUDA is available.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the best model checkpoint. If None, model is not saved.")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to save TensorBoard logs. If None, logging is disabled.")
    parser.add_argument("--cache_data", action="store_true",
                        help="Enable caching of preprocessed data to speed up loading.")

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(args.random_seed)

    print("Starting GRU-D Sepsis Prediction Training with MDN, Weighted Loss, and N-hour Horizon")
    print(f"Args: {args}")

    if args.transition_weight <= 0:
        print("Warning: transition_weight should be > 0. Setting to 1.0.")
        args.transition_weight = 1.0
    if args.transition_weight > 1.0:
        print(f"Using transition_weight: {args.transition_weight} for S_t=0 -> S_{{t+{args.prediction_horizon}}}=1 sepsis transitions.")


    file_paths = glob(os.path.join(args.data_dir, "*.psv")) + glob(os.path.join(args.data_dir, "*.csv"))
    if not file_paths:
        print(f"Error: No .psv or .csv files found in {args.data_dir}")
        return
    
    print(f"Found {len(file_paths)} patient files.")

    cache_dir_path = None
    if args.cache_data:
        cache_dir_path = os.path.join(args.data_dir, ".cache_sepsis_predictor_gru_d_mdn_weighted_horizon") # More specific cache name
        print(f"Data caching enabled. Cache directory: {cache_dir_path}")

    if args.val_split > 0 and args.val_split < 1:
        train_files, val_files = train_test_split(file_paths, test_size=args.val_split, random_state=args.random_seed)
    else:
        train_files = file_paths
        val_files = [] 
        if args.val_split != 0 :
            print("Warning: Invalid val_split value. Disabling validation split.")

    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    if not train_files:
        print("Error: No files available for training after split.")
        return
        
    print("Computing normalization statistics from training data (original features only)...")
    mean_stats, std_stats = compute_normalisation_stats(train_files)
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        mean_path = os.path.join(args.save_dir, "mean_stats.npy")
        std_path = os.path.join(args.save_dir, "std_stats.npy")
        try:
            np.save(mean_path, mean_stats); np.save(std_path, std_stats)
            print(f"Saved normalization statistics to {mean_path} and {std_path}")
        except Exception as e:
            print(f"Warning: Could not save normalization statistics: {e}")

    # Pass prediction_horizon to SepsisDataset
    train_dataset = SepsisDataset(train_files, mean_stats, std_stats, args.prediction_horizon,
                                  args.max_seq_len, args.include_current_sepsis_label, 
                                  cache_dir=cache_dir_path, dataset_type="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, 
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True if DEVICE.type == 'cuda' else False,
                              drop_last=True if len(train_dataset) > args.batch_size else False) # drop_last for stability if batch size issues
    
    val_loader = None
    if val_files:
        val_dataset = SepsisDataset(val_files, mean_stats, std_stats, args.prediction_horizon,
                                    args.max_seq_len, args.include_current_sepsis_label, 
                                    cache_dir=cache_dir_path, dataset_type="val")
        if len(val_dataset) > 0: # Only create loader if dataset is not empty
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                    num_workers=min(4, os.cpu_count() or 1), pin_memory=True if DEVICE.type == 'cuda' else False)
        else:
            print("Validation dataset is empty after processing. Skipping validation loader.")


    num_model_input_features = len(FEATURE_COLUMNS)
    if args.include_current_sepsis_label: # This flag controls if SepsisLabel_t is part of X_t input
        num_model_input_features += 1
    
    print(f"Model input feature dimension: {num_model_input_features}")
    print(f"Predicting SepsisLabel_{{t+{args.prediction_horizon}}} based on X_t.")


    model = SepsisGRUDMDN(
        input_size=num_model_input_features,
        hidden_size=args.hidden_size,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
        num_mdn_components=args.num_mdn_components
    ).to(DEVICE)

    if args.torch_compile:
        if DEVICE.type == 'mps':
            print("Warning: torch.compile() on MPS has known limitations. Skipping compilation.")
        elif DEVICE.type in ['cuda', 'cpu'] and hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
            print("Attempting to compile the model with torch.compile()...")
            try:
                model = torch.compile(model)
                print("Model compiled successfully.")
            except Exception as e:
                print(f"Warning: torch.compile() failed: {e}. Proceeding without compilation.")
        elif not (hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2):
            print("torch.compile() requires PyTorch 2.0 or later. Skipping compilation.")
        else:
            print(f"torch.compile() not attempted for device type: {DEVICE.type}")


    print(f"Model architecture: {model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) # Increased patience

    scaler = None
    if args.mixed_precision and DEVICE.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision training enabled with GradScaler.")
    elif args.mixed_precision and DEVICE.type != 'cuda':
        print("Warning: Mixed precision training requested, but CUDA is not available. Proceeding without it.")

    best_val_auc = -1.0
    best_model_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Model checkpoints will be saved in: {args.save_dir}")

    writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"TensorBoard logs will be saved in: {args.log_dir}")

    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Pass prediction_horizon to train_epoch
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, args.num_mdn_components, 
                                 args.transition_weight, args.prediction_horizon, scaler)
        
        val_loss, val_auc, val_accuracy, val_logit_variances, val_trans_recall = \
            float('nan'), float('nan'), float('nan'), np.array([]), float('nan')
        
        if val_loader: # Only evaluate if val_loader exists
            # Pass prediction_horizon to evaluate
            val_loss, val_auc, val_accuracy, val_logit_variances, val_trans_recall = evaluate(
                model, val_loader, DEVICE, args.prediction_horizon
            )
        
        epoch_duration = time.time() - start_time
        avg_val_logit_var = np.mean(val_logit_variances) if val_logit_variances.size > 0 else float('nan')
        
        val_trans_recall_str = "N/A"
        if not np.isnan(val_trans_recall):
            val_trans_recall_str = f"{val_trans_recall:.4f}"
        elif val_loader: # If val_loader exists, it means validation was attempted
             val_trans_recall_str = f"N/A (no 0->{args.prediction_horizon}h transitions in val)"
        # If val_loader is None, it remains "N/A" by default.

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Avg Val Logit Var: {avg_val_logit_var:.4f} | "
            f"Val TransRec(0->{args.prediction_horizon}h): {val_trans_recall_str} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
            if val_loader and not np.isnan(val_loss): # Check val_loader and val_loss
                writer.add_scalar('Loss/validation', val_loss, epoch + 1)
                writer.add_scalar('AUC/validation', val_auc, epoch + 1)
                writer.add_scalar('Accuracy/validation', val_accuracy, epoch + 1)
                if val_logit_variances.size > 0:
                    writer.add_scalar('Uncertainty/AvgValLogitVar', avg_val_logit_var, epoch + 1)
                    writer.add_histogram('Uncertainty/ValLogitVariances', val_logit_variances, epoch + 1)
                if not np.isnan(val_trans_recall):
                     writer.add_scalar(f'Recall/ValTransition_0_to_{args.prediction_horizon}h', val_trans_recall, epoch + 1)

        if val_loader and not np.isnan(val_loss):
            scheduler.step(val_loss)

        if args.save_dir and val_loader and not np.isnan(val_auc):
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_path = os.path.join(args.save_dir, f"best_model_horizon{args.prediction_horizon}_checkpoint.pt") # Include horizon in name
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(), # Save uncompiled model state if compiled
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_auc': best_val_auc,
                    'args': args 
                }
                torch.save(checkpoint, best_model_path)
                print(f"Epoch {epoch+1}: New best model saved to {best_model_path} with Val AUC: {val_auc:.4f}")

    
    print("Training finished.")

    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} with Val AUC: {best_val_auc:.4f}")
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        
        saved_args = checkpoint.get('args')
        if saved_args:
            num_model_input_features_loaded = len(FEATURE_COLUMNS)
            if saved_args.include_current_sepsis_label:
                num_model_input_features_loaded += 1
            
            loaded_hidden_size = saved_args.hidden_size
            loaded_num_gru_layers = saved_args.num_gru_layers
            loaded_dropout = saved_args.dropout
            loaded_num_mdn_components = saved_args.num_mdn_components
        else: # Fallback for older checkpoints without saved args
            print("Warning: Checkpoint does not contain saved arguments. Using current script arguments for model loading.")
            num_model_input_features_loaded = num_model_input_features # From current run
            loaded_hidden_size = args.hidden_size
            loaded_num_gru_layers = args.num_gru_layers
            loaded_dropout = args.dropout
            loaded_num_mdn_components = args.num_mdn_components

        model_to_load = SepsisGRUDMDN(
            input_size=num_model_input_features_loaded,
            hidden_size=loaded_hidden_size,
            num_gru_layers=loaded_num_gru_layers,
            dropout=loaded_dropout,
            num_mdn_components=loaded_num_mdn_components
        ).to(DEVICE)
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        model_for_eval = model_to_load # Use this fresh model for final eval
        
        print("Best model loaded. Performing final evaluation on validation set...")
        if val_loader:
            val_loss, val_auc, val_accuracy, val_logit_variances, val_trans_recall = evaluate(
                model_for_eval, val_loader, DEVICE, args.prediction_horizon # Use current args.prediction_horizon
            )
            avg_val_logit_var = np.mean(val_logit_variances) if val_logit_variances.size > 0 else float('nan')
            val_trans_recall_str = "N/A"
            if not np.isnan(val_trans_recall):
                val_trans_recall_str = f"{val_trans_recall:.4f}"
            elif val_loader:
                val_trans_recall_str = f"N/A (no 0->{args.prediction_horizon}h transitions in val)"

            print(
                f"  Final Val Stats (Best Model): Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_accuracy:.4f}, "
                f"Avg Logit Var: {avg_val_logit_var:.4f}, TransRec(0->{args.prediction_horizon}h): {val_trans_recall_str}"
            )
    elif args.save_dir:
        print(f"No best model was saved (or found). Skipping loading best model.")

    if writer:
        writer.close()

if __name__ == "__main__":
    main()