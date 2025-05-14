import os
import shutil
import random
# Removed pandas as it's not strictly needed for this version

def get_sepsis_label_status(file_path):
    """
    Checks if a patient file contains any SepsisLabel=1 entry.
    Returns True if SepsisLabel=1 is found, False otherwise.
    Returns None if the file can't be read or SepsisLabel column is missing.
    """
    try:
        with open(file_path, 'r') as f:
            header = f.readline().strip()
            if not header:
                return None
            column_names = header.split('|')
            try:
                sepsis_label_idx = column_names.index('SepsisLabel')
            except ValueError:
                return None

            for line in f:
                fields = line.strip().split('|')
                if len(fields) > sepsis_label_idx:
                    # No need for try-except ValueError here if we assume '1' or '0'
                    if fields[sepsis_label_idx] == '1':
                        return True
        return False
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def create_sepsis_only_dataset(source_dir, output_dir_name, expected_sepsis_cases):
    """
    Creates a dataset containing only sepsis patient files.
    """
    workspace_root = "/Users/rkph/Desktop/projects/sepsis"
    full_output_dir = os.path.join(workspace_root, output_dir_name)

    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    if os.path.exists(full_output_dir):
        print(f"Warning: Output directory '{full_output_dir}' already exists. Removing it.")
        shutil.rmtree(full_output_dir)
    os.makedirs(full_output_dir)
    print(f"Created output directory: '{full_output_dir}' for sepsis-only files.")

    sepsis_files = []
    
    all_files = [f for f in os.listdir(source_dir) if f.endswith('.psv') and os.path.isfile(os.path.join(source_dir, f))]
    
    print(f"Processing {len(all_files)} PSV files from '{source_dir}' to identify sepsis cases...")

    for i, filename in enumerate(all_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(all_files)} files...")
        
        file_path = os.path.join(source_dir, filename)
        is_sepsis = get_sepsis_label_status(file_path)
        
        if is_sepsis is True:
            sepsis_files.append(filename)
        # Non-sepsis files (is_sepsis is False) and error files (is_sepsis is None) are ignored

    print(f"Identified {len(sepsis_files)} sepsis files.")

    if expected_sepsis_cases is not None and len(sepsis_files) != expected_sepsis_cases:
        print(f"Warning: Expected approximately {expected_sepsis_cases} sepsis cases based on prior analysis, but found {len(sepsis_files)} files with SepsisLabel=1.")
    
    # Copy sepsis files
    if not sepsis_files:
        print("No sepsis files found. The output directory will be empty.")
    else:
        print(f"Copying {len(sepsis_files)} sepsis files to '{full_output_dir}'...")
        for filename in sepsis_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(full_output_dir, filename))

    total_files_in_output = len(os.listdir(full_output_dir))
    print(f"Sepsis-only dataset creation complete. '{full_output_dir}' contains {total_files_in_output} files.")

if __name__ == "__main__":
    # Configuration
    SOURCE_PATIENT_DATA_DIR = "/Users/rkph/Desktop/projects/sepsis/physionet.org/files/challenge-2019/1.0.0/training/training_setA"
    # Changed directory name slightly to reflect its content
    SEPSIS_ONLY_DATA_DIR_NAME = "sepsis_only_training_data_A" 
    # From average_frequency_calculator_output.txt: "Total sepsis cases identified: 1790"
    # This is now for informational/warning purposes if the count differs.
    EXPECTED_SEPSIS_CASES = 1790 

    create_sepsis_only_dataset(SOURCE_PATIENT_DATA_DIR, SEPSIS_ONLY_DATA_DIR_NAME, EXPECTED_SEPSIS_CASES)
    print("Script finished.") 