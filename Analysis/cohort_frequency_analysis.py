import sys
import os
from collections import defaultdict # Added for summing frequencies
import concurrent.futures # Added for multithreading
from sepsis_analysis_utils import (
    get_column_headers,
    # check_if_sepsis_case, # Removed
    # analyze_cohort, # Removed
    compare_cohorts,
    find_psv_files,
    process_patient_file_for_analysis, # Added
    print_nicely_formatted_results # Added
)

def main():
    data_directory = './physionet.org/files/challenge-2019/1.0.0/training/training_setA/'

    # Define columns to consistently drop from FEATURE FREQUENCY ANALYSIS in both cohorts
    # SepsisLabel will also be dropped from feature frequency analysis, handled below.
    base_columns_to_drop = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'Age', 'Gender']
    
    psv_files = find_psv_files(data_directory, limit=None) 
    
    if not psv_files:
        print(f"No PSV files found in {data_directory}")
        return

    example_file_path = psv_files[0]
    all_headers = get_column_headers(example_file_path)
    if not all_headers:
        print(f"Could not read headers from {example_file_path}")
        return
    
    # For cohort frequency analysis, SepsisLabel itself should not be a reported feature.
    # process_patient_file_for_analysis will still use SepsisLabel internally from all_headers to determine 'is_sepsis'.
    analysis_columns_to_drop = base_columns_to_drop + ['SepsisLabel']
    print(f"Analyzing features. Dropping from frequency reports: {', '.join(analysis_columns_to_drop)}")

    # Headers for which frequencies will actually be reported in tables.
    headers_for_freq_reporting = [h for h in all_headers if h not in analysis_columns_to_drop]

    # Initialize accumulators for sepsis and non-sepsis cohorts
    sum_sepsis_freqs = defaultdict(float)
    sepsis_files_with_data_count = 0
    total_sepsis_patient_files = 0 # Total files identified as sepsis cases

    sum_non_sepsis_freqs = defaultdict(float)
    non_sepsis_files_with_data_count = 0
    total_non_sepsis_patient_files = 0 # Total files identified as non-sepsis
    processed_count_for_progress = 0 # For progress reporting

    print(f"\nProcessing {len(psv_files)} files from {data_directory} for single-pass cohort analysis using multiple threads...")

    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    if not psv_files:
        print(f"No files provided to analyze.") # Should be caught by earlier check, but good for safety
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(process_patient_file_for_analysis, file, all_headers, analysis_columns_to_drop): file for file in psv_files}
            
            total_files_to_process = len(psv_files)
            print(f"Submitted {total_files_to_process} files for processing to {num_workers} worker threads.")

            for future in concurrent.futures.as_completed(future_to_file):
                file_path_key = future_to_file[future]
                processed_count_for_progress += 1
                try:
                    file_analysis_results = future.result()

                    if file_analysis_results['rows'] > 0:
                        if file_analysis_results['is_sepsis']:
                            total_sepsis_patient_files += 1
                            sepsis_files_with_data_count += 1
                            for feature, freq in file_analysis_results['frequencies'].items():
                                sum_sepsis_freqs[feature] += freq
                        else:
                            total_non_sepsis_patient_files += 1
                            non_sepsis_files_with_data_count += 1
                            for feature, freq in file_analysis_results['frequencies'].items():
                                sum_non_sepsis_freqs[feature] += freq
                    
                    if processed_count_for_progress % (total_files_to_process // 10 if total_files_to_process >=10 else 1) == 0 or processed_count_for_progress == total_files_to_process:
                        print(f"Cohort Analysis: Processed {processed_count_for_progress}/{total_files_to_process} files...")
                
                except FileNotFoundError:
                    print(f"Warning: File not found {file_path_key} (skipped). Cohort Analysis.")
                except Exception as exc:
                    print(f"Error processing file {os.path.basename(file_path_key)} for Cohort Analysis: {exc} (skipped)")

    print(f"\nFound {total_sepsis_patient_files} sepsis patient files and {total_non_sepsis_patient_files} non-sepsis patient files.")

    # Calculate average frequencies for Sepsis Cohort
    avg_sepsis_freqs = {}
    if sepsis_files_with_data_count > 0:
        for feature in headers_for_freq_reporting:
            avg_sepsis_freqs[feature] = sum_sepsis_freqs[feature] / sepsis_files_with_data_count
    else:
        for feature in headers_for_freq_reporting:
            avg_sepsis_freqs[feature] = 0.0
    
    # Print results for Sepsis Cohort
    if total_sepsis_patient_files > 0:
        print_nicely_formatted_results(headers_for_freq_reporting, avg_sepsis_freqs, sepsis_files_with_data_count, "SEPSIS")
    else:
        print("\nNo sepsis patient files found or processed with data; cannot display Sepsis cohort frequencies.")

    # Calculate average frequencies for Non-Sepsis Cohort
    avg_non_sepsis_freqs = {}
    if non_sepsis_files_with_data_count > 0:
        for feature in headers_for_freq_reporting:
            avg_non_sepsis_freqs[feature] = sum_non_sepsis_freqs[feature] / non_sepsis_files_with_data_count
    else:
        for feature in headers_for_freq_reporting:
            avg_non_sepsis_freqs[feature] = 0.0

    # Print results for Non-Sepsis Cohort
    if total_non_sepsis_patient_files > 0:
        print_nicely_formatted_results(headers_for_freq_reporting, avg_non_sepsis_freqs, non_sepsis_files_with_data_count, "NON-SEPSIS")
    else:
        print("\nNo non-sepsis patient files found or processed with data; cannot display Non-Sepsis cohort frequencies.")

    # Compare the cohorts
    if total_sepsis_patient_files > 0 or total_non_sepsis_patient_files > 0:
        compare_cohorts(avg_sepsis_freqs, avg_non_sepsis_freqs, total_sepsis_patient_files, total_non_sepsis_patient_files)
    else:
        print("\nNo data processed for either cohort; cannot perform comparison.")

if __name__ == '__main__':
    main() 