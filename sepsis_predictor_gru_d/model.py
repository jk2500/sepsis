#!/usr/bin/env python
"""
Train a GRU-D model to predict the next‑timestep sepsis label from PhysioNet Sepsis Challenge data.

Usage (example):
    python grud_sepsis.py --data_dir /path/to/training_data --epochs 30 --batch_size 64 --lr 1e-3 \
                          --include_current_sepsis_label --transition_weight 5.0

Expecting each patient record in CSV/PSV format with a column named "SepsisLabel" and the standard feature set
from the 2019 PhysioNet/Computing in Cardiology Challenge.
"""
import argparse
import os
from glob import glob
import time # For timing epochs

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split # For train/val split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
DEFAULT_TRANSITION_WEIGHT = 1.0 # New default: no extra weight

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
TIME_COLUMN = "Time"  # hours since ICU admission (implicit in files; generated below)

# ---------------------------
# Utility functions
# ---------------------------

def load_patient_file(path: str) -> pd.DataFrame:
    """Load a .psv / .csv file into a DataFrame and add a Time column."""
    df = pd.read_csv(path, sep="|" if path.endswith(".psv") else ",")
    df[TIME_COLUMN] = np.arange(len(df)) # Time in hours, assuming 1 row = 1 hour
    return df


def compute_normalisation_stats(file_paths):
    """Compute mean and std for each feature across the training set (ignoring NaNs)."""
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
    def __init__(self, file_paths, mean, std, max_seq_len: int | None = None, include_current_sepsis_label: bool = False):
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len
        self.mean = mean 
        self.std = std   
        self.num_original_features = len(FEATURE_COLUMNS)
        self.include_current_sepsis_label = include_current_sepsis_label # For feature input
        # Note: We will always return current_y for potential use in weighted loss,
        # regardless of whether it's also an *input feature*.

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        df = load_patient_file(self.file_paths[idx])
        if self.max_seq_len is not None:
            df = df.iloc[: self.max_seq_len]
            
        x_original = df[FEATURE_COLUMNS].values.astype(np.float32)
        y_sepsis_labels_current_t = df[LABEL_COLUMN].values.astype(np.float32) # SepsisLabel_t

        seq_len = x_original.shape[0]

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

        if self.include_current_sepsis_label: # If SepsisLabel_t is to be an *input feature*
            sepsis_label_t_feature = y_sepsis_labels_current_t.reshape(-1, 1) 
            x_model_input = np.concatenate([x_norm_imputed_zeros, sepsis_label_t_feature], axis=1)
            m_sepsis_label = np.ones((seq_len, 1), dtype=np.float32)
            m_model_input = np.concatenate([m_original, m_sepsis_label], axis=1)
            delta_sepsis_label = np.zeros((seq_len, 1), dtype=np.float32)
            delta_model_input = np.concatenate([delta_original, delta_sepsis_label], axis=1)
            x_last_sepsis_label = sepsis_label_t_feature 
            x_last_model_input = np.concatenate([x_last_obsv_norm_original, x_last_sepsis_label], axis=1)

        # Target: Next‑step label (SepsisLabel_{t+1})
        if len(y_sepsis_labels_current_t) > 0:
            y_next_t = np.concatenate([y_sepsis_labels_current_t[1:], np.array([y_sepsis_labels_current_t[-1]], dtype=np.float32)])
        else: 
            y_next_t = np.array([], dtype=np.float32)

        return {
            "x": torch.from_numpy(x_model_input),
            "m": torch.from_numpy(m_model_input),
            "delta": torch.from_numpy(delta_model_input),
            "x_last": torch.from_numpy(x_last_model_input),
            "y_current": torch.from_numpy(y_sepsis_labels_current_t), # SepsisLabel_t
            "y_next": torch.from_numpy(y_next_t),                  # SepsisLabel_{t+1}
            "length": seq_len, 
        }
    

def collate_fn(batch):
    batch.sort(key=lambda b: b["length"], reverse=True)
    lengths = [b["length"] for b in batch]
    max_len = lengths[0] if lengths else 0

    def pad_tensor(tensors_list, is_label_or_length_or_current_y):
        # For y_current, y_next (batch, seq_len)
        if is_label_or_length_or_current_y: 
            return torch.stack([torch.nn.functional.pad(t, (0, max_len - len(t))) for t in tensors_list])
        # For x, m, delta, x_last (batch, seq_len, num_features)
        else: 
            return torch.stack([torch.nn.functional.pad(t, (0, 0, 0, max_len - t.size(0))) for t in tensors_list])

    x = pad_tensor([b["x"] for b in batch], False)
    m = pad_tensor([b["m"] for b in batch], False)
    delta = pad_tensor([b["delta"] for b in batch], False)
    x_last = pad_tensor([b["x_last"] for b in batch], False)
    y_current = pad_tensor([b["y_current"] for b in batch], True) # Pad y_current
    y_next = pad_tensor([b["y_next"] for b in batch], True)       # Pad y_next (target)
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    # Return y_current as well
    return x, m, delta, x_last, y_current, y_next, lengths_tensor


# ---------------------------
# GRU‑D Model Components
# ... (GRUDCell and GRUD remain unchanged) ...
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
            if hasattr(layer, 'weight'):
                 nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                 nn.init.zeros_(layer.bias)
        
        nn.init.zeros_(self.gamma_x_decay)
        nn.init.zeros_(self.gamma_h_decay)

    def forward(self, x_t, m_t, delta_t, x_last_obsv_norm, h_prev):
        gamma_x_t = torch.exp(-torch.relu(self.gamma_x_decay) * delta_t)
        x_hat_t = m_t * x_t + (1 - m_t) * (gamma_x_t * x_last_obsv_norm)

        gamma_h_t_factor = torch.exp(-torch.relu(self.gamma_h_decay))
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
        if dropout > 0.0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(dropout)

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
                
                h_new_for_cell, _ = self.cells[l](x_t_layer_input, 
                                                  m_t_layer_input, 
                                                  delta_t_layer_input, 
                                                  x_last_t_layer_input, 
                                                  h_prev_for_cell)
                
                h_layer_states[l] = h_new_for_cell
                x_t_layer_input = h_new_for_cell

                if self.dropout_layer and l < self.num_layers - 1:
                    x_t_layer_input = self.dropout_layer(x_t_layer_input)
            
            outputs_from_last_layer_seq.append(h_layer_states[-1])

        outputs_stacked = torch.stack(outputs_from_last_layer_seq, dim=1)
        return outputs_stacked, h_layer_states

# ---------------------------
# Main Sepsis Prediction Model
# ... (SepsisGRUDRNN remains unchanged) ...
class SepsisGRUDRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_gru_layers=1, dropout=0.0, output_size=1):
        super().__init__()
        self.grud = GRUD(input_size, hidden_size, num_gru_layers, dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, m, delta, x_last, lengths=None):
        grud_output_seq, _ = self.grud(x, m, delta, x_last, lengths)
        logits = self.fc(grud_output_seq)
        
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits

# ---------------------------
# Training and Evaluation
# ---------------------------
def train_epoch(model, dataloader, optimizer, criterion_unreduced, device, transition_weight: float):
    model.train()
    total_loss = 0.0
    num_valid_timesteps_for_loss_avg = 0 # Count of elements contributing to loss

    # criterion_unreduced should be nn.BCEWithLogitsLoss(reduction='none')
    
    for x, m, delta, x_last, y_current, y_next, lengths in dataloader:
        x, m, delta, x_last, y_current, y_next, lengths = (
            x.to(device), m.to(device), delta.to(device), 
            x_last.to(device), y_current.to(device), y_next.to(device), lengths.to(device)
        )
        
        optimizer.zero_grad()
        predictions_logits = model(x, m, delta, x_last, lengths) # (batch, seq_len)
        
        # Mask for loss calculation: y_next[t] is original y[t+1].
        # Valid timesteps for loss: 0 to L-2 for original length L.
        loss_mask = torch.zeros_like(y_next, dtype=torch.bool, device=device)
        for i, l_val in enumerate(lengths):
            if l_val > 1: # Need at least 2 timesteps for one y_next label
                 loss_mask[i, :l_val - 1] = True
        
        if loss_mask.sum() == 0: # No valid timesteps in batch
            continue

        # Calculate unreduced loss for all elements
        unreduced_loss = criterion_unreduced(predictions_logits, y_next) # (batch, seq_len)

        # Create sample_weights tensor
        sample_weights = torch.ones_like(y_next, device=device) # Default weight is 1
        
        if transition_weight > 1.0:
            # Identify 0 -> 1 transitions
            # y_current is SepsisLabel_t, y_next is SepsisLabel_{t+1}
            # We care about y_current[t] == 0 and y_next[t] == 1 (which corresponds to SepsisLabel_{t+1})
            # The loss_mask applies to predictions and y_next.
            # So, we need y_current corresponding to the valid y_next elements.
            is_0_to_1_transition = (y_current == 0) & (y_next == 1)
            sample_weights[is_0_to_1_transition] = transition_weight
        
        # Apply weights and mask to the loss
        # Only consider elements where loss_mask is True
        weighted_loss = unreduced_loss * sample_weights
        final_loss_for_batch = weighted_loss[loss_mask].sum() / loss_mask.sum() # Average over valid timesteps
        
        final_loss_for_batch.backward()
        optimizer.step()
        
        # total_loss accumulates sum of losses, num_valid_timesteps counts elements
        total_loss += weighted_loss[loss_mask].sum().item() 
        num_valid_timesteps_for_loss_avg += loss_mask.sum().item()

    return total_loss / num_valid_timesteps_for_loss_avg if num_valid_timesteps_for_loss_avg > 0 else 0.0


def evaluate(model, dataloader, criterion_eval, device, include_current_sepsis_label_arg: bool):
    # criterion_eval should be nn.BCEWithLogitsLoss() with default reduction='mean'
    model.eval()
    total_loss = 0.0
    num_valid_timesteps_for_loss_avg = 0
    all_true_next_labels_list = [] 
    all_pred_next_probs_list = []  
    all_current_labels_list = []   

    with torch.no_grad():
        # In evaluate, y_current is from the dataloader, not necessarily from x's features
        for x, m, delta, x_last, y_current, y_next, lengths in dataloader: 
            x, m, delta, x_last, y_current, y_next, lengths = (
                x.to(device), m.to(device), delta.to(device), 
                x_last.to(device), y_current.to(device), y_next.to(device), lengths.to(device)
            )

            predictions_logits = model(x, m, delta, x_last, lengths)
            
            loss_mask = torch.zeros_like(y_next, dtype=torch.bool, device=device)
            for i, l_val in enumerate(lengths):
                 if l_val > 1:
                    loss_mask[i, :l_val - 1] = True
            
            if loss_mask.sum() == 0:
                continue

            # For evaluation loss, use the standard (unweighted) criterion
            loss = criterion_eval(predictions_logits[loss_mask], y_next[loss_mask])
            total_loss += loss.item() * loss_mask.sum().item()
            num_valid_timesteps_for_loss_avg += loss_mask.sum().item()

            probs = torch.sigmoid(predictions_logits[loss_mask])
            all_true_next_labels_list.extend(y_next[loss_mask].cpu().numpy())
            all_pred_next_probs_list.extend(probs.cpu().numpy())
            
            # Store current SepsisLabel_t (y_current from dataloader) for transition metric
            # y_current[loss_mask] aligns with y_next[loss_mask] and preds[loss_mask]
            all_current_labels_list.extend(y_current[loss_mask].cpu().numpy())
            
    avg_loss = total_loss / num_valid_timesteps_for_loss_avg if num_valid_timesteps_for_loss_avg > 0 else 0.0
    
    if not all_true_next_labels_list:
        return avg_loss, 0.0, 0.0, np.nan

    np_true_next_labels = np.array(all_true_next_labels_list)
    np_pred_next_probs = np.array(all_pred_next_probs_list)
    
    try:
        auc = roc_auc_score(np_true_next_labels, np_pred_next_probs)
    except ValueError:
        auc = 0.0 
        
    accuracy = accuracy_score(np_true_next_labels, (np_pred_next_probs > 0.5).astype(int))

    transition_recall_0_to_1 = np.nan 
    # Now use all_current_labels_list which directly comes from y_current
    if all_current_labels_list: # Check if list is not empty
        np_current_labels = np.array(all_current_labels_list)
        
        actual_0_to_1_mask = (np_current_labels == 0) & (np_true_next_labels == 1)
        num_actual_0_to_1_transitions = np.sum(actual_0_to_1_mask)

        if num_actual_0_to_1_transitions > 0:
            preds_for_0_to_1_transitions = np_pred_next_probs[actual_0_to_1_mask]
            num_correctly_predicted_0_to_1 = np.sum(preds_for_0_to_1_transitions > 0.5)
            transition_recall_0_to_1 = num_correctly_predicted_0_to_1 / num_actual_0_to_1_transitions
    
    return avg_loss, auc, accuracy, transition_recall_0_to_1


# ---------------------------
# Main script
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train GRU-D for Sepsis Prediction")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing patient CSV/PSV files")
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
    parser.add_argument("--transition_weight", type=float, default=DEFAULT_TRANSITION_WEIGHT,
                        help="Weight for 0->1 sepsis transitions in the loss function. "
                             "Default 1.0 (no special weighting).")


    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(args.random_seed)

    print("Starting GRU-D Sepsis Prediction Training")
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
    
    # SepsisDataset now takes include_current_sepsis_label for feature engineering
    # but always provides y_current for potential loss weighting.
    train_dataset = SepsisDataset(train_files, mean_stats, std_stats, args.max_seq_len, args.include_current_sepsis_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=min(2, os.cpu_count() or 1), pin_memory=True if DEVICE.type == 'cuda' else False)
    
    val_loader = None
    if val_files:
        val_dataset = SepsisDataset(val_files, mean_stats, std_stats, args.max_seq_len, args.include_current_sepsis_label)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=min(2, os.cpu_count() or 1), pin_memory=True if DEVICE.type == 'cuda' else False)

    num_model_input_features = len(FEATURE_COLUMNS)
    if args.include_current_sepsis_label: # This refers to using it as a MODEL INPUT
        num_model_input_features += 1
    
    print(f"Model input feature dimension: {num_model_input_features}")

    model = SepsisGRUDRNN(
        input_size=num_model_input_features,
        hidden_size=args.hidden_size,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
        output_size=1 
    ).to(DEVICE)

    print(f"Model architecture: {model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    # For training with weighted loss
    criterion_train = nn.BCEWithLogitsLoss(reduction='none') 
    # For evaluation (standard, unweighted loss)
    criterion_eval = nn.BCEWithLogitsLoss() # Default reduction='mean'

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Pass transition_weight to train_epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion_train, DEVICE, args.transition_weight)
        
        val_loss, val_auc, val_accuracy, val_trans_recall = float('nan'), float('nan'), float('nan'), float('nan')
        if val_loader:
            # Pass include_current_sepsis_label only for how evaluate extracts current label info for metric.
            # evaluate uses y_current from dataloader now, so this arg is less critical for it.
            val_loss, val_auc, val_accuracy, val_trans_recall = evaluate(model, val_loader, criterion_eval, DEVICE, args.include_current_sepsis_label)
        
        epoch_duration = time.time() - start_time
        
        val_trans_recall_str = "N/A"
        if not np.isnan(val_trans_recall):
            val_trans_recall_str = f"{val_trans_recall:.4f}"
        elif val_loader:
            # The logic for N/A here is simpler now because y_current is always available in eval
            val_trans_recall_str = "N/A (no 0->1 transitions in val set)"


        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val TransRec(0->1): {val_trans_recall_str} | "
            f"Duration: {epoch_duration:.2f}s"
        )
    
    print("Training finished.")

if __name__ == "__main__":
    main()