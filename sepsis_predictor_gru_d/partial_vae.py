import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialVAE(nn.Module):
    def __init__(self, feature_dim, hidden_dim_grud, latent_dim, 
                 encoder_hidden_dims=[128, 64], decoder_hidden_dims=[64, 128]):
        super(PartialVAE, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim_grud = hidden_dim_grud
        self.latent_dim = latent_dim

        # Encoder
        # Input: masked_features (feature_dim), mask (feature_dim), grud_hidden_state (hidden_dim_grud)
        # Total input dim for encoder: feature_dim + feature_dim + hidden_dim_grud
        current_dim = feature_dim * 2 + hidden_dim_grud
        encoder_layers = []
        for h_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = h_dim
        self.encoder_mlp = nn.Sequential(*encoder_layers)
        self.fc_z_mean = nn.Linear(current_dim, latent_dim)
        self.fc_z_logvar = nn.Linear(current_dim, latent_dim)

        # Decoder
        # Input: latent_sample (latent_dim), grud_hidden_state (hidden_dim_grud)
        # Total input dim for decoder: latent_dim + hidden_dim_grud
        current_dim_dec = latent_dim + hidden_dim_grud
        decoder_layers = []
        for h_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(current_dim_dec, h_dim))
            decoder_layers.append(nn.ReLU())
            current_dim_dec = h_dim
        decoder_layers.append(nn.Linear(current_dim_dec, feature_dim)) # Output reconstructed feature means
        self.decoder_mlp = nn.Sequential(*decoder_layers)

    def encode(self, x_t, m_t, h_grud_t):
        # Apply mask to input features (observed features are kept, unobserved are zeroed)
        masked_x_t = x_t * m_t 
        
        # Concatenate masked features, mask, and GRU-D hidden state
        # Ensure h_grud_t is correctly shaped if it's 1D per sample in batch
        if h_grud_t.ndim == 1:
            h_grud_t = h_grud_t.unsqueeze(0) # If single sample, make it batch-like
        if h_grud_t.ndim == 2 and x_t.ndim == 2 and h_grud_t.shape[0] != x_t.shape[0]: # Batch processing
             h_grud_t = h_grud_t.expand(x_t.shape[0], -1)


        combined_input = torch.cat([masked_x_t, m_t, h_grud_t], dim=1)
        
        hidden_e = self.encoder_mlp(combined_input)
        z_mean = self.fc_z_mean(hidden_e)
        z_logvar = self.fc_z_logvar(hidden_e)
        return z_mean, z_logvar

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z_t, h_grud_t):
         # Ensure h_grud_t is correctly shaped
        if h_grud_t.ndim == 1:
            h_grud_t = h_grud_t.unsqueeze(0)
        if h_grud_t.ndim == 2 and z_t.ndim == 2 and h_grud_t.shape[0] != z_t.shape[0]: # Batch processing
            h_grud_t = h_grud_t.expand(z_t.shape[0], -1)
            
        combined_input_dec = torch.cat([z_t, h_grud_t], dim=1)
        x_prime_mu = self.decoder_mlp(combined_input_dec) # Outputs means of reconstructed features
        return x_prime_mu

    def forward(self, x_t, m_t, h_grud_t):
        z_mean, z_logvar = self.encode(x_t, m_t, h_grud_t)
        z_t = self.reparameterize(z_mean, z_logvar)
        x_prime_mu = self.decode(z_t, h_grud_t)
        return x_prime_mu, z_mean, z_logvar

    def loss_function(self, x_t, m_t, x_prime_mu, z_mean, z_logvar, beta=1.0):
        # Reconstruction Loss (MSE for observed features only)
        # Only penalize reconstruction error for features that were actually observed
        recon_loss = F.mse_loss(x_prime_mu * m_t, x_t * m_t, reduction='sum') / m_t.sum()
        # Alternative if reduction='sum' is too large, consider 'mean' on non-zero elements
        # Or calculate MSE only on elements where m_t is 1
        # num_observed = m_t.sum()
        # if num_observed > 0:
        #    recon_loss = F.mse_loss(x_prime_mu[m_t == 1], x_t[m_t == 1], reduction='mean')
        # else:
        #    recon_loss = torch.tensor(0.0, device=x_t.device)


        # KL Divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        # Normalize KLD by batch size (or sum over batch as well, check common practices)
        kld_loss = kld_loss / x_t.size(0) # Average KLD per sample in batch

        total_loss = recon_loss + beta * kld_loss
        return total_loss, recon_loss, kld_loss

    def sample_conditional(self, x_obs_t, m_obs_t, h_grud_t, num_samples=1):
        """
        Samples missing features given observed features and GRU-D hidden state.
        For active inference, x_obs_t will contain the true observed values, 
        and m_obs_t will mark these. For features to be imputed, their
        corresponding entries in x_obs_t can be zero (as they'll be masked anyway by m_obs_t
        if we use encode directly) or some placeholder. The key is m_obs_t.

        This function generates K full feature vectors where missing values are imputed.
        Args:
            x_obs_t (Tensor): Current timestep's feature vector, potentially with placeholders for missing.
                              Shape (batch_size, feature_dim) or (feature_dim)
            m_obs_t (Tensor): Mask for x_obs_t (1 if observed, 0 if missing/to be imputed).
                              Shape (batch_size, feature_dim) or (feature_dim)
            h_grud_t (Tensor): GRU-D hidden state. Shape (batch_size, hidden_dim_grud) or (hidden_dim_grud)
            num_samples (int): Number of samples to generate.
        Returns:
            Tensor: Sampled complete feature vectors. Shape (num_samples, batch_size, feature_dim) 
                    or (num_samples, feature_dim) if batch_size was 1.
        """
        if x_obs_t.ndim == 1: # Single instance
            x_obs_t = x_obs_t.unsqueeze(0)
            m_obs_t = m_obs_t.unsqueeze(0)
            if h_grud_t.ndim ==1: # ensure h_grud_t is also batched if x_obs_t was
                 h_grud_t = h_grud_t.unsqueeze(0)
            single_instance_mode = True
        else: # Already batched
            single_instance_mode = False

        batch_size = x_obs_t.shape[0]
        
        # Encode observed data to get latent distribution parameters
        # For features to be imputed, x_obs_t values don't matter as they are masked by m_obs_t in encode
        z_mean, z_logvar = self.encode(x_obs_t, m_obs_t, h_grud_t)

        # For sampling, we might want to tile z_mean and z_logvar if num_samples > 1
        # Or sample `num_samples` times in a loop if memory is a concern for large num_samples
        
        # Tile z_mean and z_logvar for multi-sampling
        z_mean_expanded = z_mean.unsqueeze(0).expand(num_samples, batch_size, self.latent_dim)
        z_logvar_expanded = z_logvar.unsqueeze(0).expand(num_samples, batch_size, self.latent_dim)
        
        # Flatten for reparameterization if it expects (N, latent_dim)
        z_mean_flat = z_mean_expanded.reshape(-1, self.latent_dim)
        z_logvar_flat = z_logvar_expanded.reshape(-1, self.latent_dim)
        
        sampled_z_flat = self.reparameterize(z_mean_flat, z_logvar_flat)
        sampled_z = sampled_z_flat.reshape(num_samples, batch_size, self.latent_dim)

        # Expand h_grud_t for decoding
        h_grud_t_expanded = h_grud_t.unsqueeze(0).expand(num_samples, batch_size, self.hidden_dim_grud)
        
        # Flatten for decoder
        sampled_z_dec_flat = sampled_z.reshape(-1, self.latent_dim)
        h_grud_t_dec_flat = h_grud_t_expanded.reshape(-1, self.hidden_dim_grud)

        # Decode to get the mean of the reconstructed (and imputed) features
        reconstructed_x_mu_flat = self.decode(sampled_z_dec_flat, h_grud_t_dec_flat)
        reconstructed_x_mu = reconstructed_x_mu_flat.reshape(num_samples, batch_size, self.feature_dim)

        if single_instance_mode:
            return reconstructed_x_mu.squeeze(1) # Shape (num_samples, feature_dim)
        else:
            return reconstructed_x_mu # Shape (num_samples, batch_size, feature_dim)

# Example Usage (Illustrative - won't run without data and GRU-D):
if __name__ == '__main__':
    # Define dimensions (example)
    feat_dim = 40  # From FEATURE_COLUMNS
    h_dim_gru = 64 # Example GRU-D hidden size
    lat_dim = 20   # Latent dimension for VAE

    # Instantiate the Partial VAE
    p_vae = PartialVAE(feature_dim=feat_dim, hidden_dim_grud=h_dim_gru, latent_dim=lat_dim)

    # Dummy data for a single timestep and single patient
    # (In practice, this would come from your SepsisDataset and GRU-D model)
    dummy_x_t = torch.randn(1, feat_dim)         # Patient features at timestep t
    dummy_m_t = torch.ones(1, feat_dim)          # Mask (all observed for this example)
    dummy_m_t[0, 5:10] = 0                       # Artificially make some features missing
    dummy_h_grud = torch.randn(1, h_dim_gru)     # GRU-D hidden state

    # Forward pass
    x_reconstructed_mu, z_mu, z_logvar = p_vae(dummy_x_t, dummy_m_t, dummy_h_grud)
    print("Reconstructed x_mu shape:", x_reconstructed_mu.shape)
    print("Latent z_mu shape:", z_mu.shape)

    # Calculate loss
    total_loss, recon_loss, kld = p_vae.loss_function(dummy_x_t, dummy_m_t, x_reconstructed_mu, z_mu, z_logvar)
    print(f"Total Loss: {total_loss.item()}, Recon Loss: {recon_loss.item()}, KLD: {kld.item()}")

    # Test conditional sampling
    # Suppose we want to impute the features that were set to missing (indices 5-9)
    # x_obs_t would be dummy_x_t (or a version where true values for 5-9 are unknown)
    # m_obs_t would be the mask where indices 5-9 are 0, others are 1
    m_for_sampling = torch.ones(1, feat_dim)
    m_for_sampling[0, 5:10] = 0 # These are the features we want the VAE to help impute
    x_for_sampling = dummy_x_t.clone() # True values, encoder will use m_for_sampling

    num_generated_samples = 5
    sampled_features = p_vae.sample_conditional(x_for_sampling, m_for_sampling, dummy_h_grud, num_samples=num_generated_samples)
    print(f"Shape of {num_generated_samples} conditionally sampled full feature vectors:", sampled_features.shape)
    # Each row in sampled_features[:, 0, :] is a full feature vector.
    # The values at indices 5-9 in these vectors are the VAE's imputed samples for those missing features.
    # The values at other indices are reconstructions of the observed features.
    
    # For actual use in active inference, you'd pick one missing feature f at a time.
    # You would then take the f-th column from `sampled_features` (across num_generated_samples)
    # as your K samples for that specific feature x_f.
    # e.g., if feature index 5 was missing:
    # samples_for_feature_5 = sampled_features[:, 0, 5] # (num_generated_samples,)

    print("\nExample of imputed values for feature 5 (if it was missing):")
    if m_for_sampling[0,5] == 0:
        print(sampled_features[:,0,5]) 