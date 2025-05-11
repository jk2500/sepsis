# Plan for Conditional Generative Model (Inspired by EDDI Partial VAE)

## 1. Objective

To develop a conditional generative model capable of imputing missing physiological features in time-series sepsis patient data. This model will replace the current placeholder sampling in `inference_active.py` and enable a more principled, information-theoretic approach to active feature acquisition, aligning with methods like EDDI. The primary goal is to accurately estimate $p(x_f | X_{\text{obs}})$ for any missing feature $x_f$ given observed features $X_{\text{obs}}$, and use this to calculate Expected Information Gain (EIG).

## 2. Model Choice: Partial VAE (adapted for Time Series)

We will implement a Variational Autoencoder (VAE) adapted to handle partially observed data (a "Partial VAE"). The EDDI repository (`p_vae.py`) provides a strong reference for this, particularly its handling of input masks and modification of the VAE objective.

**Justification**:
-   **Principled Imputation**: VAEs can learn complex data distributions and generate realistic samples.
-   **Conditional Generation**: They can be structured to generate samples for missing features conditioned on observed ones.
-   **EDDI Precedent**: The EDDI paper and codebase demonstrate the effectiveness of Partial VAEs for active information acquisition.

**Adaptation for Time Series**:
The EDDI VAE processes static data points. For our sepsis use case with GRU-D, $X_{\text{obs}}$ is a sequence. We need to consider how to incorporate temporal context:
-   **Option A (Simpler)**: Train a VAE on individual timesteps (slices of patient data). The conditioning $X_{\text{obs}}$ for imputing $x_f$ at time $t$ would be the observed features *at time $t$*. Past information could be incorporated by feeding the GRU-D hidden state at time $t-1$ (or $t$) as an additional conditioning input to the VAE's encoder and decoder.
-   **Option B (More Complex)**: Develop a Recurrent VAE (e.g., VAE with GRU/LSTM components in its encoder/decoder) that natively models sequences and can be conditioned on partial sequences.

We will start with **Option A** for feasibility, using the GRU-D hidden state as context.

## 3. Key Components (PyTorch Implementation)

The model will be implemented in PyTorch to integrate with the existing Sepsis GRU-D predictor.

### 3.1. Encoder

-   **Input**:
    -   `x_t`: Feature vector at timestep $t$ (from `FEATURE_COLUMNS`).
    -   `m_t`: Mask vector for `x_t` (1 if observed, 0 if missing).
    -   `h_grud_t`: (Optional, for Option A) Hidden state from GRU-D at timestep $t$ (or $t-1$) as additional context.
-   **Architecture**: Multi-layer Perceptron (MLP).
    -   Inspired by `p_vae.py`: The EDDI encoder processes `x_aug = tf.concat([self.x_flat, self.x_flat * self.F, self.b], 1)` where `x_flat` is the input, `F` and `b` are learnable parameters, and this is done per feature. Then, `self.encoded * self.mask_on_hidden` is summed.
    -   We can adapt this: For each feature $x_{t,i}$, create an embedding. If the feature is missing (masked), a special "missing" embedding could be used, or the contribution zeroed out. Concatenate these processed feature embeddings (and `h_grud_t` if used) before passing to further MLP layers.
-   **Output**: Parameters (mean $\mu_z$ and log-variance $\log\sigma_z^2$) of the latent variable distribution $q(z_t | x_t, m_t, [h_{grud_t}])$.

### 3.2. Reparameterization Trick

-   Sample latent vector $z_t = \mu_z + \sigma_z \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.

### 3.3. Decoder

-   **Input**: Sampled latent vector $z_t$ (and `h_grud_t` if used for conditioning decoder too).
-   **Architecture**: MLP.
-   **Output**: Parameters of the distribution for each reconstructed feature $x'_{t,i}$. Since our features are continuous and normalized, this will likely be the mean $\mu_{x'_{t,i}}$ of a Gaussian distribution. The variance can be fixed or also predicted. The EDDI decoder (`fc_uci_decoder` in `codings.py`) outputs means, assuming fixed observation noise for Gaussian or parameters for Bernoulli.

### 3.4. Loss Function

The VAE loss will be the sum of reconstruction loss and KL divergence:
$L = \mathbb{E}_{q(z_t|x_t,m_t)}[\log p(x_t|z_t, m_t)] - \beta \cdot KL(q(z_t|x_t,m_t) || p(z_t))$

-   **Reconstruction Loss**:
    -   Crucially, this should only be calculated for the *observed* features based on the input mask $m_t$.
    -   If predicting Gaussian parameters: Negative log-likelihood of observed $x_{t,i}$ given $\mu_{x'_{t,i}}$ (and $\sigma_{x'_{t,i}}$).
    -   EDDI's `_gaussian_log_likelihood` calculates `0.5 * tf.square(targets - mean) / tf.square(std) + tf.log(std)`. It applies the mask *before* calling this: `self.x * self.mask`, `self.decoded * self.mask`.
-   **KL Divergence**: $KL(q(z_t|x_t,m_t) || p(z_t))$, where $p(z_t)$ is the prior (standard Gaussian $\mathcal{N}(0, I)$).
-   **$\beta$**: Annealing parameter for KL term (optional, can start with $\beta=1$).

### 3.5. Conditional Sampling Function for Inference $p(x_{t,f} | X_{t, obs}, [h_{grud_t}])$

This is the core function needed for active feature acquisition. To sample a missing feature $x_{t,f}$ given observed features $X_{t, obs}$ (represented by $x_t$ and $m_t$ where $m_{t,f}=0$):

1.  **Encode Observed**: Pass $x_t$ (with $x_{t,f}$ typically zeroed out or handled by masking in encoder) and $m_t$ (and $h_{grud_t}$) through the trained encoder to get $\mu_z, \sigma_z$.
2.  **Sample Latent**: Draw $K$ samples $z_t^{(k)}$ from $q(z_t | X_{t, obs}, [h_{grud_t}])$.
3.  **Decode Samples**: Pass each $z_t^{(k)}$ (and $h_{grud_t}$) through the trained decoder to get $K$ sets of reconstructed feature parameters (e.g., means $\mu_{x'}^{(k)}$).
4.  **Extract Target Feature**: From each reconstructed set, take the value(s) corresponding to the target missing feature $x_{t,f}$. These $K$ values, $\{\mu_{x'_{t,f}}^{(1)}, ..., \mu_{x'_{t,f}}^{(K)}\}$, approximate samples from $p(x_{t,f} | X_{t, obs}, [h_{grud_t}])$.

The EDDI repo's `completion` function (`p_vae.py`) and its usage in `active_learning_functions.py` (`im = completion(x, mask, M, vae)`) serve this purpose.

## 4. Data Considerations

-   **Input Features**: The `FEATURE_COLUMNS` from `model_MDN.py`.
-   **Normalization**: Use the same mean/std normalization as the main GRU-D model. The EDDI code also preprocesses data (e.g., scaling to [0,1]). We should stick to our existing normalization.
-   **Training Data**: The same training dataset used for the GRU-D model. The VAE will be trained on individual timesteps, potentially augmented with GRU-D hidden states.
-   **Handling Missingness during VAE Training**:
    -   Use the actual missingness patterns from the data.
    -   Additionally, EDDI uses "artificial missingness" (`args.p` in `main_active_learning.py`) during training, where it randomly drops out *already observed* features. This makes the VAE more robust at imputing various patterns. We should adopt this.

## 5. Integration with `sepsis_predictor_gru_d/inference_active.py`

-   The `hypothetical_normalized_samples` list will be replaced.
-   For each `feature_model_idx` corresponding to a missing feature:
    -   Call the new VAE's conditional sampling function (described in 3.5) to get $K$ samples for that feature, conditioned on the *current* patient's observed data (`x_patient_seq[input_timestep_idx, :]`, `m_patient_seq[input_timestep_idx, :]`) and potentially the GRU-D hidden state at that point.
    -   The loop `for sample_val_tensor in hypothetical_normalized_samples:` will iterate over these $K$ new samples.

## 6. Technology Stack

-   **PyTorch**: To align with the existing GRU-D Sepsis predictor.

## 7. Development Steps/Phases

1.  **VAE Base Implementation (PyTorch)**:
    -   Define Encoder, Decoder, Reparameterization trick.
    -   Implement VAE loss (Reconstruction + KL).
    -   Train on a simple, complete dataset first to verify correctness.
2.  **Adapt to Partial VAE**:
    -   Modify Encoder to accept and use input masks.
    -   Modify Reconstruction Loss to only consider observed features.
    -   Implement artificial missingness during training.
    -   Train on sepsis timesteps (ignoring temporal context initially if simpler).
3.  **(Conditional Context - Option A)**:
    -   Modify VAE (Encoder & Decoder) to accept GRU-D hidden state as additional conditioning input.
    -   Adapt training: requires running GRU-D to get hidden states for each timestep of VAE training data.
4.  **Implement Conditional Sampling Function**:
    -   Develop the function as described in section 3.5.
5.  **Training and Evaluation**:
    -   Train the Partial VAE (with context if implemented) on the sepsis dataset.
    -   Evaluate imputation quality (e.g., on held-out observed values).
6.  **Integration**:
    -   Integrate the conditional sampling function into `simulate_active_feature_acquisition` in `inference_active.py`.
7.  **End-to-End Testing & Refinement**:
    -   Test the full active inference pipeline.
    -   Compare EIG results with the previous placeholder method.

## 8. Open Questions & Challenges

-   **Temporal Context Handling**: The chosen "Option A" (using GRU-D hidden state) is a heuristic. True sequential generative modeling (Option B) is much harder. We need to assess if Option A provides sufficient conditional power.
-   **Complexity of EDDI's Encoder Input**: EDDI's `x_aug = tf.concat([self.x_flat, self.x_flat * self.F, self.b], 1)` and feature-wise processing before aggregation is specific. We need to decide if a simpler MLP encoder that takes the masked feature vector directly (perhaps with specific handling for masked inputs) is sufficient or if this more complex feature-wise embedding and aggregation is key.
-   **Computational Cost**: Sampling $K$ times from the VAE for *each* candidate missing feature at *each* step of active acquisition can be computationally intensive. $K$ (EDDI uses `M=50` for imputation) will need tuning.
-   **Stability and Training**: VAEs can be tricky to train. Careful hyperparameter tuning will be needed.
-   **Evaluation of Generative Model**: Defining good metrics for how well $p(x_f | X_{\text{obs}})$ is modeled, beyond just reconstruction error of $X_{\text{obs}}$.

This plan provides a roadmap. Details will be refined during implementation. 