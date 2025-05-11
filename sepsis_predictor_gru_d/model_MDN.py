#!/usr/bin/env python
"""
Train a GRU-D model with an MDN head to predict the next‑timestep sepsis label 
and its uncertainty from PhysioNet Sepsis Challenge data, with weighted loss for transitions.

Usage (example):
    python grud_sepsis_mdn.py --data_dir /path/to/training_data --epochs 30 --batch_size 64 --lr 1e-3 \
                              --num_mdn_components 3 --transition_weight 5.0

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
    "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]
LABEL_COLUMN = "SepsisLabel"
TIME_COLUMN = "Time"

# ---------------------------
# Utility functions
# ---------------------------

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
    def __init__(self, file_paths, mean, std, max_seq_len: int | None = None, 
                 include_current_sepsis_label: bool = False, 
                 cache_dir: str | None = None, dataset_type: str = "train"): 
        self.mean = mean 
        self.std = std   
        self.num_original_features = len(FEATURE_COLUMNS)
        self.max_seq_len = max_seq_len
        self.include_current_sepsis_label = include_current_sepsis_label
        self.items = []

        cache_attempted = False
        if cache_dir:
            cache_attempted = True
            os.makedirs(cache_dir, exist_ok=True)
            
            relevant_args_str = f"max_seq_len={self.max_seq_len}_include_label={self.include_current_sepsis_label}"
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
                    
                    cached_max_seq_len = cached_meta.get('args_snapshot', {}).get('max_seq_len')
                    cached_include_label = cached_meta.get('args_snapshot', {}).get('include_current_sepsis_label')

                    if (cached_num_files == expected_num_files and
                        cached_max_seq_len == self.max_seq_len and
                        cached_include_label == self.include_current_sepsis_label):
                        self.items = cached_content['data']
                        print(f"Successfully loaded {len(self.items)} items for {dataset_type} from cache.")
                    else:
                        print("Cache metadata mismatch (num_files or critical args). Re-processing.")
                except Exception as e:
                    print(f"Error loading from cache: {e}. Re-processing.")
        
        if not self.items: 
            if cache_attempted:
                 print(f"Processing {dataset_type} data and saving to cache: {getattr(self, 'cache_path', 'in-memory')}")
            else:
                 print(f"Processing {dataset_type} data (caching disabled).")
            
            self.items = self._process_all_files(file_paths)
            
            if cache_attempted and hasattr(self, 'cache_path') and self.items: 
                metadata_to_save = {
                    'num_files': len(file_paths),
                    'args_snapshot': { 
                        'max_seq_len': self.max_seq_len,
                        'include_current_sepsis_label': self.include_current_sepsis_label
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
            processed_items.append(self._get_single_item_data_from_file(file_path))
        return processed_items

    def _get_single_item_data_from_file(self, file_path): 
        df = load_patient_file(file_path)
        if self.max_seq_len is not None:
            df = df.iloc[: self.max_seq_len]
            
        x_original = df[FEATURE_COLUMNS].values.astype(np.float32)
        y_sepsis_labels_current_t = df[LABEL_COLUMN].values.astype(np.float32) 

        seq_len = x_original.shape[0]
        if seq_len == 0: 
            num_input_features = self.num_original_features + (1 if self.include_current_sepsis_label else 0)
            return {
                "x": torch.empty((0, num_input_features), dtype=torch.float32),
                "m": torch.empty((0, num_input_features), dtype=torch.float32),
                "delta": torch.empty((0, num_input_features), dtype=torch.float32),
                "x_last": torch.empty((0, num_input_features), dtype=torch.float32),
                "y_current": torch.empty(0, dtype=torch.float32),
                "y_next": torch.empty(0, dtype=torch.float32),
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
            sepsis_label_t_feature = y_sepsis_labels_current_t.reshape(-1, 1) 
            x_model_input = np.concatenate([x_norm_imputed_zeros, sepsis_label_t_feature], axis=1)
            m_sepsis_label = np.ones((seq_len, 1), dtype=np.float32)
            m_model_input = np.concatenate([m_original, m_sepsis_label], axis=1)
            delta_sepsis_label = np.zeros((seq_len, 1), dtype=np.float32)
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
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_fn(batch):
    batch.sort(key=lambda b: b["length"], reverse=True)
    lengths = [b["length"] for b in batch]
    max_len = lengths[0] if lengths else 0

    # Determine num_features dynamically from the first item if batch is not empty
    # This assumes all items in a batch will have the same feature dimension for 'x'
    # (which they should, based on SepsisDataset logic).
    # If include_current_sepsis_label is True, one more feature is added.
    num_x_features = len(FEATURE_COLUMNS)
    if batch and "x" in batch[0] and batch[0]["x"].ndim > 1: # Check if x is 2D (seq_len, features)
        num_x_features = batch[0]["x"].size(1)


    if not batch:
        # Use the dynamically determined or default num_x_features for empty batch handling
        return (torch.empty((0, 0, num_x_features)), torch.empty((0, 0, num_x_features)), 
                torch.empty((0, 0, num_x_features)), torch.empty((0, 0, num_x_features)), 
                torch.empty((0, 0)), torch.empty((0,0)), torch.tensor([], dtype=torch.long))


    def pad_tensor(tensors_list, is_label_or_length_or_current_y):
        if is_label_or_length_or_current_y:
            return torch.stack([torch.nn.functional.pad(t, (0, max_len - len(t))) for t in tensors_list])
        else:
            return torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_len - t.size(0))) for t in tensors_list])

    x = pad_tensor([b["x"] for b in batch], False)
    m = pad_tensor([b["m"] for b in batch], False)
    delta = pad_tensor([b["delta"] for b in batch], False)
    x_last = pad_tensor([b["x_last"] for b in batch], False)
    y_current = pad_tensor([b["y_current"] for b in batch], True)
    y_next = pad_tensor([b["y_next"] for b in batch], True)
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return x, m, delta, x_last, y_current, y_next, lengths_tensor

# ---------------------------
# GRU‑D Model Components
# ---------------------------
class GRUDCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma_x_decay = nn.Parameter(torch.Tensor(input_size))
        self.gamma_h_decay = nn.Parameter(torch.Tensor(hidden_size))
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
        gamma_x_t = torch.exp(-torch.relu(self.gamma_x_decay) * delta_t)
        x_hat_t = m_t * x_t + (1 - m_t) * (gamma_x_t * x_last_obsv_norm)
        
        # Aggregate delta_t for hidden state decay (e.g., by taking the mean across features)
        # This delta_t_for_h will have shape (batch_size, 1)
        delta_t_for_h = torch.mean(delta_t, dim=1, keepdim=True)
        
        # self.gamma_h_decay has shape (hidden_size,)
        # delta_t_for_h has shape (batch_size, 1)
        # Their product will broadcast to (batch_size, hidden_size)
        gamma_h_t_factor = torch.exp(-torch.relu(self.gamma_h_decay) * delta_t_for_h)
        h_prev_decayed = gamma_h_t_factor * h_prev
        
        concat_for_gates = torch.cat([x_hat_t, m_t, gamma_x_t], dim=1)
        r_t = torch.sigmoid(self.W_r(concat_for_gates) + self.U_r(h_prev_decayed))
        z_t = torch.sigmoid(self.W_z(concat_for_gates) + self.U_z(h_prev_decayed))
        h_tilde_t = torch.tanh(self.W_h_tilde(concat_for_gates) + self.U_h_tilde(r_t * h_prev_decayed))
        h_curr = (1 - z_t) * h_prev_decayed + z_t * h_tilde_t
        return h_curr, x_hat_t

class GRUD(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
            x_t_layer_input = x[:, t, :]
            m_t_layer_input = m[:, t, :]
            delta_t_layer_input = delta[:, t, :]
            x_last_t_layer_input = x_last[:, t, :]
            for l in range(self.num_layers):
                h_prev_for_cell = h_layer_states[l]
                if l > 0: 
                    m_t_layer_input = torch.ones_like(x_t_layer_input)
                    delta_t_layer_input = torch.zeros_like(x_t_layer_input)
                    x_last_t_layer_input = x_t_layer_input
                h_new_for_cell, _ = self.cells[l](x_t_layer_input, m_t_layer_input, delta_t_layer_input, x_last_t_layer_input, h_prev_for_cell)
                h_layer_states[l] = h_new_for_cell
                x_t_layer_input = h_new_for_cell
                if self.dropout_layer and l < self.num_layers - 1: x_t_layer_input = self.dropout_layer(x_t_layer_input)
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
        self.fc_mdn = nn.Linear(hidden_size, num_mdn_components * 2)

    def forward(self, x, m, delta, x_last, lengths=None):
        grud_output_seq, _ = self.grud(x, m, delta, x_last, lengths)
        mdn_params = self.fc_mdn(grud_output_seq)
        pi_logits, mus = torch.split(mdn_params, self.num_mdn_components, dim=-1)
        pis = F.softmax(pi_logits, dim=-1)
        return pis, mus

# ---------------------------
# MDN Loss Function
# ---------------------------
def mdn_loss_bernoulli(pis, mus, targets, sample_weights=None):
    targets_expanded = targets.unsqueeze(-1) 
    log_prob_y_given_component_k = -(F.binary_cross_entropy_with_logits(
        mus, targets_expanded.expand_as(mus), reduction='none'
    ))
    log_pis = torch.log(pis + 1e-8)
    weighted_log_probs = log_pis + log_prob_y_given_component_k
    log_likelihood = torch.logsumexp(weighted_log_probs, dim=-1) # (batch, seq_len)
    
    nll_per_timestep = -log_likelihood # (batch, seq_len)

    if sample_weights is not None:
        nll_per_timestep = nll_per_timestep * sample_weights # Apply weights

    return nll_per_timestep # Return unreduced, weighted NLL

# ---------------------------
# Training and Evaluation
# ---------------------------
def train_epoch(model, dataloader, optimizer, device, num_mdn_components, transition_weight: float, scaler=None): # Added scaler
    model.train()
    total_loss_sum = 0.0 # Sum of losses over valid timesteps
    num_valid_timesteps_for_loss = 0

    for x, m, delta, x_last, y_current, y_next, lengths in dataloader: # Added y_current
        x, m, delta, x_last, y_current, y_next, lengths = (
            x.to(device), m.to(device), delta.to(device), 
            x_last.to(device), y_current.to(device), y_next.to(device), lengths.to(device)
        )
        
        optimizer.zero_grad(set_to_none=True) # Changed for mixed precision recommendation

        if scaler: # Mixed precision
            with torch.cuda.amp.autocast():
                pis, mus = model(x, m, delta, x_last, lengths)
                loss_mask = torch.zeros_like(y_next, dtype=torch.bool, device=device)
                for i, l_val in enumerate(lengths):
                    if l_val > 1:
                         loss_mask[i, :l_val - 1] = True
                
                if loss_mask.sum() == 0:
                    continue

                sample_weights = torch.ones_like(y_next, device=device)
                if transition_weight > 1.0:
                    is_0_to_1_transition = (y_current == 0) & (y_next == 1)
                    sample_weights[is_0_to_1_transition] = transition_weight
                
                nll_per_timestep_weighted = mdn_loss_bernoulli(pis, mus, y_next, sample_weights)
                loss = nll_per_timestep_weighted[loss_mask].mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Grad clipping
            scaler.step(optimizer)
            scaler.update()
        else: # Standard precision
            pis, mus = model(x, m, delta, x_last, lengths) 
            
            loss_mask = torch.zeros_like(y_next, dtype=torch.bool, device=device)
            for i, l_val in enumerate(lengths):
                if l_val > 1:
                     loss_mask[i, :l_val - 1] = True
            
            if loss_mask.sum() == 0:
                continue

            sample_weights = torch.ones_like(y_next, device=device)
            if transition_weight > 1.0:
                is_0_to_1_transition = (y_current == 0) & (y_next == 1)
                sample_weights[is_0_to_1_transition] = transition_weight
            
            nll_per_timestep_weighted = mdn_loss_bernoulli(pis, mus, y_next, sample_weights)
            loss = nll_per_timestep_weighted[loss_mask].mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Grad clipping
            optimizer.step()
        
        total_loss_sum += nll_per_timestep_weighted[loss_mask].sum().item()
        num_valid_timesteps_for_loss += loss_mask.sum().item()

    return total_loss_sum / num_valid_timesteps_for_loss if num_valid_timesteps_for_loss > 0 else 0.0


def evaluate(model, dataloader, device): # Removed num_mdn_components, Removed include_current_sepsis_label_arg
    model.eval()
    total_loss_sum_eval = 0.0 # Sum of unweighted losses for evaluation
    num_valid_timesteps_for_loss_eval = 0
    all_true_next_labels_list = []
    all_pred_next_probs_list = []
    all_logit_variances_list = []
    all_current_labels_list = [] # For transition metric

    with torch.no_grad():
        for x, m, delta, x_last, y_current, y_next, lengths in dataloader: # Added y_current
            x, m, delta, x_last, y_current, y_next, lengths = (
                x.to(device), m.to(device), delta.to(device), 
                x_last.to(device), y_current.to(device), y_next.to(device), lengths.to(device)
            )

            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    pis, mus = model(x, m, delta, x_last, lengths)
            else:
                pis, mus = model(x, m, delta, x_last, lengths)
            
            loss_mask = torch.zeros_like(y_next, dtype=torch.bool, device=device)
            for i, l_val in enumerate(lengths):
                 if l_val > 1:
                    loss_mask[i, :l_val - 1] = True
            
            if loss_mask.sum() == 0:
                continue

            # For evaluation, calculate unweighted NLL
            # No autocast here as nll_per_timestep_eval uses logsumexp etc which should be stable in float32
            nll_per_timestep_eval = mdn_loss_bernoulli(pis, mus, y_next, sample_weights=None) # (batch, seq_len)
            loss_eval = nll_per_timestep_eval[loss_mask].mean()
            total_loss_sum_eval += nll_per_timestep_eval[loss_mask].sum().item()
            num_valid_timesteps_for_loss_eval += loss_mask.sum().item()

            probs_k = torch.sigmoid(mus)
            mean_probs = torch.sum(pis * probs_k, dim=-1)
            
            expected_logit = torch.sum(pis * mus, dim=-1)
            expected_logit_sq = torch.sum(pis * (mus.pow(2)), dim=-1)
            logit_variance = torch.relu(expected_logit_sq - expected_logit.pow(2))

            all_true_next_labels_list.extend(y_next[loss_mask].cpu().numpy())
            all_pred_next_probs_list.extend(mean_probs[loss_mask].cpu().numpy())
            all_logit_variances_list.extend(logit_variance[loss_mask].cpu().numpy())
            all_current_labels_list.extend(y_current[loss_mask].cpu().numpy()) # Use y_current from dataloader
            
    avg_loss_eval = total_loss_sum_eval / num_valid_timesteps_for_loss_eval if num_valid_timesteps_for_loss_eval > 0 else 0.0
    
    if not all_true_next_labels_list:
        return avg_loss_eval, 0.0, 0.0, np.array([]), np.nan # Added np.nan for transition recall

    np_true_next_labels = np.array(all_true_next_labels_list)
    np_pred_next_probs = np.array(all_pred_next_probs_list)

    try:
        auc = roc_auc_score(np_true_next_labels, np_pred_next_probs)
    except ValueError:
        auc = 0.0
        
    accuracy = accuracy_score(np_true_next_labels, (np_pred_next_probs > 0.5).astype(int))
    
    transition_recall_0_to_1 = np.nan
    if all_current_labels_list:
        np_current_labels = np.array(all_current_labels_list)
        actual_0_to_1_mask = (np_current_labels == 0) & (np_true_next_labels == 1)
        num_actual_0_to_1_transitions = np.sum(actual_0_to_1_mask)
        if num_actual_0_to_1_transitions > 0:
            preds_for_0_to_1_transitions = np_pred_next_probs[actual_0_to_1_mask]
            num_correctly_predicted_0_to_1 = np.sum(preds_for_0_to_1_transitions > 0.5)
            transition_recall_0_to_1 = num_correctly_predicted_0_to_1 / num_actual_0_to_1_transitions
            
    return avg_loss_eval, auc, accuracy, np.array(all_logit_variances_list), transition_recall_0_to_1


# ---------------------------
# Main script
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train GRU-D with MDN for Sepsis Prediction with weighted loss")
    parser.add_argument("--data_dir", type=str, default="physionet.org/files/challenge-2019/1.0.0/training/training_setA", help="Directory containing patient CSV/PSV files")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE, help="GRU-D hidden size")
    parser.add_argument("--num_gru_layers", type=int, default=DEFAULT_NUM_GRU_LAYERS, help="Number of GRU-D layers")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate between GRU-D layers")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data for validation (0 to disable)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--include_current_sepsis_label", action="store_true", 
                        help="Include SepsisLabel_t as an input feature to the model.")
    parser.add_argument("--num_mdn_components", type=int, default=DEFAULT_NUM_MDN_COMPONENTS, help="Number of MDN mixture components")
    parser.add_argument("--transition_weight", type=float, default=DEFAULT_TRANSITION_WEIGHT, # New argument
                        help="Weight for 0->1 sepsis transitions in the loss function. Default 1.0 (no special weighting).")
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

    print("Starting GRU-D Sepsis Prediction Training with MDN and Weighted Loss")
    print(f"Args: {args}")

    if args.transition_weight <= 0:
        print("Warning: transition_weight should be > 0. Setting to 1.0.")
        args.transition_weight = 1.0
    if args.transition_weight > 1.0:
        print(f"Using transition_weight: {args.transition_weight} for 0->1 sepsis transitions.")


    file_paths = glob(os.path.join(args.data_dir, "*.psv")) + glob(os.path.join(args.data_dir, "*.csv"))
    if not file_paths:
        print(f"Error: No .psv or .csv files found in {args.data_dir}")
        return
    
    print(f"Found {len(file_paths)} patient files.")

    cache_dir_path = None
    if args.cache_data:
        cache_dir_path = os.path.join(args.data_dir, ".cache_sepsis_predictor_gru_d")
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
    
    # Save normalization stats if a save directory is provided
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True) # Ensure directory exists
        mean_path = os.path.join(args.save_dir, "mean_stats.npy")
        std_path = os.path.join(args.save_dir, "std_stats.npy")
        try:
            np.save(mean_path, mean_stats)
            np.save(std_path, std_stats)
            print(f"Saved normalization statistics to {mean_path} and {std_path}")
        except Exception as e:
            print(f"Warning: Could not save normalization statistics: {e}")

    train_dataset = SepsisDataset(train_files, mean_stats, std_stats, args.max_seq_len, args.include_current_sepsis_label, 
                                  cache_dir=cache_dir_path, dataset_type="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, 
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True if DEVICE.type == 'cuda' else False)
    
    val_loader = None
    if val_files:
        val_dataset = SepsisDataset(val_files, mean_stats, std_stats, args.max_seq_len, args.include_current_sepsis_label, 
                                    cache_dir=cache_dir_path, dataset_type="val")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                num_workers=min(4, os.cpu_count() or 1), pin_memory=True if DEVICE.type == 'cuda' else False)

    num_model_input_features = len(FEATURE_COLUMNS)
    if args.include_current_sepsis_label:
        num_model_input_features += 1
    
    print(f"Model input feature dimension: {num_model_input_features}")

    model = SepsisGRUDMDN(
        input_size=num_model_input_features,
        hidden_size=args.hidden_size,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
        num_mdn_components=args.num_mdn_components
    ).to(DEVICE)

    if args.torch_compile:
        if DEVICE.type == 'mps':
            print("Warning: torch.compile() is enabled but the device is MPS. MPS backend for torch.compile() can have limitations (e.g., exceeding constant buffer limits). Skipping compilation. Consider running without --torch_compile on MPS if issues persist or for potentially more stable execution.")
        elif DEVICE.type == 'cuda' or DEVICE.type == 'cpu': # Add CPU just in case, though benefits are more on GPU
            print("Attempting to compile the model with torch.compile()...")
            try:
                if hasattr(torch, '__version__') and int(torch.__version__.split('.')[0]) >= 2:
                    model = torch.compile(model)
                    print("Model compiled successfully.")
                else:
                    print("torch.compile() requires PyTorch 2.0 or later. Skipping compilation.")
            except Exception as e:
                print(f"Warning: torch.compile() failed: {e}. Proceeding without compilation.")
        else:
            print(f"torch.compile() not attempted for device type: {DEVICE.type}")

    print(f"Model architecture: {model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    scaler = None
    if args.mixed_precision and DEVICE.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision training enabled with GradScaler.")
    elif args.mixed_precision and DEVICE.type != 'cuda':
        print("Warning: Mixed precision training (--mixed_precision) was requested, but CUDA is not available. Proceeding without it.")

    best_val_auc = -1.0  # Initialize with a value lower than any possible AUC
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
        
        # Pass transition_weight to train_epoch
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, args.num_mdn_components, args.transition_weight, scaler) # Pass scaler
        
        val_loss, val_auc, val_accuracy, val_logit_variances, val_trans_recall = float('nan'), float('nan'), float('nan'), np.array([]), float('nan')
        if val_loader:
            val_loss, val_auc, val_accuracy, val_logit_variances, val_trans_recall = evaluate(
                model, val_loader, DEVICE
            )
        
        epoch_duration = time.time() - start_time
        avg_val_logit_var = np.mean(val_logit_variances) if val_logit_variances.size > 0 else float('nan')
        
        val_trans_recall_str = "N/A"
        if not np.isnan(val_trans_recall):
            val_trans_recall_str = f"{val_trans_recall:.4f}"
        elif val_loader:
            val_trans_recall_str = "N/A (no 0->1 transitions in val set)"

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Avg Val Logit Var: {avg_val_logit_var:.4f} | "
            f"Val TransRec(0->1): {val_trans_recall_str} | " # New metric
            f"Duration: {epoch_duration:.2f}s"
        )

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
            if val_loader and not np.isnan(val_loss):
                writer.add_scalar('Loss/validation', val_loss, epoch + 1)
                writer.add_scalar('AUC/validation', val_auc, epoch + 1)
                writer.add_scalar('Accuracy/validation', val_accuracy, epoch + 1)
                if val_logit_variances.size > 0:
                    writer.add_scalar('Uncertainty/AvgValLogitVar', avg_val_logit_var, epoch + 1)
                    writer.add_histogram('Uncertainty/ValLogitVariances', val_logit_variances, epoch + 1)
                if not np.isnan(val_trans_recall):
                     writer.add_scalar('Recall/ValTransition_0_to_1', val_trans_recall, epoch + 1)

        if val_loader and not np.isnan(val_loss): # Step scheduler based on val_loss
            scheduler.step(val_loss)

        if args.save_dir and val_loader and not np.isnan(val_auc):
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_path = os.path.join(args.save_dir, "best_model_checkpoint.pt")
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
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
        
        # Adjust model instantiation if args affecting architecture were different
        # For this script, we assume the current args define the same architecture
        # as the one saved, or we create a new model based on saved_args if needed.
        # Here, we will load into the existing model structure.
        # If args.include_current_sepsis_label or other architectural args changed,
        # this might require care. For simplicity, we assume current model is compatible.
        
        # Re-initialize model to ensure it's clean before loading state_dict
        # This is important if torch.compile was used, as compiled model state might be tricky.
        # However, if torch.compile was used, loading state_dict into a fresh, uncompiled model is standard.
        
        # Determine input size from saved args for robustness
        saved_args = checkpoint.get('args')
        if saved_args:
            num_model_input_features_loaded = len(FEATURE_COLUMNS)
            if saved_args.include_current_sepsis_label:
                num_model_input_features_loaded += 1
        else: # Fallback if args not in checkpoint (older checkpoints)
            num_model_input_features_loaded = model.grud.input_size


        # Create a fresh instance of the model for loading
        # This avoids issues if 'model' was compiled
        model_to_load = SepsisGRUDMDN(
            input_size=num_model_input_features_loaded,
            hidden_size=args.hidden_size if not saved_args else saved_args.hidden_size, # Use current or saved
            num_gru_layers=args.num_gru_layers if not saved_args else saved_args.num_gru_layers,
            dropout=args.dropout if not saved_args else saved_args.dropout,
            num_mdn_components=args.num_mdn_components if not saved_args else saved_args.num_mdn_components
        ).to(DEVICE)
        
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        model = model_to_load # Replace the current model with the loaded one
        
        # Optionally, load optimizer state if further training or fine-tuning is intended
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("Best model loaded. Performing final evaluation on validation set...")
        if val_loader:
            # Run evaluate with the loaded best model
            val_loss, val_auc, val_accuracy, val_logit_variances, val_trans_recall = evaluate(
                model, val_loader, DEVICE
            )
            avg_val_logit_var = np.mean(val_logit_variances) if val_logit_variances.size > 0 else float('nan')
            val_trans_recall_str = "N/A"
            if not np.isnan(val_trans_recall):
                val_trans_recall_str = f"{val_trans_recall:.4f}"
            elif val_loader: # Added this to be consistent
                val_trans_recall_str = "N/A (no 0->1 transitions in val set)"

            print(
                f"  Final Val Stats (Best Model): Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_accuracy:.4f}, "
                f"Avg Logit Var: {avg_val_logit_var:.4f}, TransRec(0->1): {val_trans_recall_str}"
            )
    elif args.save_dir:
        print(f"No best model was saved (or found at {best_model_path}). Skipping loading.")

    if writer:
        writer.close()

    # The example inference section will be moved to a separate script.

if __name__ == "__main__":
    main()