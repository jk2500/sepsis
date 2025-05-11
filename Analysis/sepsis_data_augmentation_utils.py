import numpy as np
from collections import Counter

# It's good practice to ensure the NegativeSelectionAlgorithm class is accessible
# For now, this utility assumes it's used in an environment where NSA is available
# or that the NSA model object passed to it is self-contained.

def augment_with_nsa_detectors(X_train_scaled, y_train, nsa_model, verbose=True):
    """
    Augments training data using detectors from a trained Negative Selection Algorithm.

    Args:
        X_train_scaled (np.ndarray): The scaled training features.
        y_train (np.ndarray): The original training labels.
        nsa_model (NegativeSelectionAlgorithm): A *fitted* instance of the NegativeSelectionAlgorithm.
        verbose (bool): If True, prints information about the augmentation process.

    Returns:
        tuple: (X_train_augmented, y_train_augmented)
               - X_train_augmented (np.ndarray): Training features augmented with NSA detectors.
               - y_train_augmented (np.ndarray): Training labels augmented for NSA detectors.
               Returns original X_train_scaled, y_train if no detectors are generated.
    """
    if verbose:
        print("\n--- Attempting Data Augmentation with NSA Detectors ---")

    detectors = nsa_model.detectors_
    
    if detectors is None or detectors.shape[0] == 0:
        if verbose:
            print("No detectors available from NSA model. No data augmentation will be performed.")
        return X_train_scaled, y_train

    num_new_sepsis_samples = detectors.shape[0]
    if verbose:
        print(f"Using {num_new_sepsis_samples} generated detectors as pseudo sepsis data points.")

    # Detectors are assumed to be sepsis samples (label 1)
    detector_labels = np.ones(num_new_sepsis_samples, dtype=int)

    # Concatenate original scaled training data with the detectors
    X_train_augmented = np.vstack((X_train_scaled, detectors))
    y_train_augmented = np.concatenate((y_train, detector_labels))

    if verbose:
        print(f"Original X_train_scaled shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
        print(f"Original training target distribution: {Counter(y_train)}")
        print(f"Augmented X_train_augmented shape: {X_train_augmented.shape}, y_train_augmented shape: {y_train_augmented.shape}")
        print(f"Augmented training target distribution: {Counter(y_train_augmented)}")
        print("--- End of Data Augmentation with NSA Detectors ---\n")
    
    return X_train_augmented, y_train_augmented 