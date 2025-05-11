import os
from collections import defaultdict
import concurrent.futures # Added for multithreading
from sepsis_analysis_utils import (
    get_column_headers,
    find_psv_files,
    print_entry_level_comparison_table
)

# Define feature categories (consistent with other scripts)
VITAL_SIGNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_TESTS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
             'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
             'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
             'Fibrinogen', 'Platelets']
DEMOGRAPHICS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

ALL_CATEGORIZED_FEATURES = VITAL_SIGNS + LAB_TESTS + DEMOGRAPHICS

def _process_file_for_entry_level_analysis(file_path, features_to_analyze_in_file):
    """
    Processes a single file for entry-level analysis.
    Returns counts and totals for SepsisLabel=0 and SepsisLabel=1 entries.
    Helper for multithreaded version.
    """
    local_sepsis_0_feature_counts = defaultdict(int)
    local_sepsis_1_feature_counts = defaultdict(int)
    local_sepsis_0_total_entries = 0
    local_sepsis_1_total_entries = 0

    try:
        with open(file_path, 'r') as f:
            current_file_headers = f.readline().strip().split('|')
            try:
                current_sepsis_label_idx = current_file_headers.index('SepsisLabel')
            except ValueError:
                # print(f"Warning: 'SepsisLabel' not found in {file_path}. Skipping this file for entry analysis.")
                # Return empty results for this file if SepsisLabel is missing in its specific header
                return {
                    's0_counts': local_sepsis_0_feature_counts,
                    's1_counts': local_sepsis_1_feature_counts,
                    's0_entries': local_sepsis_0_total_entries,
                    's1_entries': local_sepsis_1_total_entries,
                    'error': f"SepsisLabel not in {os.path.basename(file_path)}"
                }

            for line_content in f:
                values = line_content.strip().split('|')
                
                if len(values) <= current_sepsis_label_idx:
                    continue 
                
                is_sepsis_entry = (values[current_sepsis_label_idx] == '1')

                if is_sepsis_entry:
                    local_sepsis_1_total_entries += 1
                else:
                    local_sepsis_0_total_entries += 1

                for header_idx, header_name in enumerate(current_file_headers):
                    if header_name in features_to_analyze_in_file: 
                        if header_idx < len(values) and values[header_idx].lower() != 'nan' and values[header_idx].strip() != '':
                            if is_sepsis_entry:
                                local_sepsis_1_feature_counts[header_name] += 1
                            else:
                                local_sepsis_0_feature_counts[header_name] += 1
    except FileNotFoundError:
         return {
            's0_counts': local_sepsis_0_feature_counts, 's1_counts': local_sepsis_1_feature_counts,
            's0_entries': 0, 's1_entries': 0, 'error': f"File not found: {os.path.basename(file_path)}"
        }
    except Exception as e:
        return {
            's0_counts': local_sepsis_0_feature_counts, 's1_counts': local_sepsis_1_feature_counts,
            's0_entries': 0, 's1_entries': 0, 'error': f"Error processing {os.path.basename(file_path)}: {e}"
        }
    
    return {
        's0_counts': local_sepsis_0_feature_counts,
        's1_counts': local_sepsis_1_feature_counts,
        's0_entries': local_sepsis_0_total_entries,
        's1_entries': local_sepsis_1_total_entries,
        'error': None
    }

def main():
    data_directory = './physionet.org/files/challenge-2019/1.0.0/training/training_setA/'
    columns_to_drop_from_analysis = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'Age', 'Gender']
    
    psv_files = find_psv_files(data_directory, limit=None) 
    
    if not psv_files:
        print(f"No PSV files found in {data_directory}")
        return

    example_file_path = psv_files[0]
    all_headers_from_example = get_column_headers(example_file_path)
    if not all_headers_from_example:
        print(f"Could not read headers from example file {example_file_path}")
        return
        
    # Check for SepsisLabel in the example file's headers to proceed.
    # Individual files will still check their own headers in the thread.
    try:
        all_headers_from_example.index('SepsisLabel')
    except ValueError:
        print("Error: 'SepsisLabel' column not found in example file headers. Cannot proceed.")
        return

    # features_to_analyze is determined once based on the example headers
    features_to_analyze = [h for h in all_headers_from_example if h != 'SepsisLabel' and h not in columns_to_drop_from_analysis]

    if columns_to_drop_from_analysis:
        print(f"The following columns will be excluded from frequency counts (in addition to SepsisLabel): {', '.join(columns_to_drop_from_analysis)}")

    # Aggregated counts and totals
    aggregated_sepsis_0_feature_counts = defaultdict(int)
    aggregated_sepsis_1_feature_counts = defaultdict(int)
    total_sepsis_0_entries = 0
    total_sepsis_1_entries = 0
    processed_count_for_progress = 0

    print(f"\nProcessing {len(psv_files)} files for entry-level analysis using multiple threads...")
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4
    if not psv_files:
        print("No files to process.")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(_process_file_for_entry_level_analysis, file, features_to_analyze): file for file in psv_files}
            
            total_files_to_process = len(psv_files)
            print(f"Submitted {total_files_to_process} files for processing to {num_workers} worker threads.")

            for future in concurrent.futures.as_completed(future_to_file):
                file_path_key = future_to_file[future]
                processed_count_for_progress += 1
                try:
                    result = future.result()
                    if result.get('error'):
                        print(f"Warning for {os.path.basename(file_path_key)}: {result['error']} (file data may be incomplete or skipped)")
                    
                    for feature, count in result['s0_counts'].items():
                        aggregated_sepsis_0_feature_counts[feature] += count
                    for feature, count in result['s1_counts'].items():
                        aggregated_sepsis_1_feature_counts[feature] += count
                    total_sepsis_0_entries += result['s0_entries']
                    total_sepsis_1_entries += result['s1_entries']

                    if processed_count_for_progress % (total_files_to_process // 10 if total_files_to_process >=10 else 1) == 0 or processed_count_for_progress == total_files_to_process:
                        print(f"Entry-Level Analysis: Processed {processed_count_for_progress}/{total_files_to_process} files...")
                except Exception as exc:
                     print(f"Critical error processing future for file {os.path.basename(file_path_key)}: {exc}")

    # Calculate frequencies from aggregated counts
    sepsis_0_frequencies = {}
    if total_sepsis_0_entries > 0:
        for feature in features_to_analyze: 
            sepsis_0_frequencies[feature] = aggregated_sepsis_0_feature_counts.get(feature, 0) / total_sepsis_0_entries
    else:
        for feature in features_to_analyze: sepsis_0_frequencies[feature] = 0.0

    sepsis_1_frequencies = {}
    if total_sepsis_1_entries > 0:
        for feature in features_to_analyze: 
            sepsis_1_frequencies[feature] = aggregated_sepsis_1_feature_counts.get(feature, 0) / total_sepsis_1_entries
    else:
        for feature in features_to_analyze: sepsis_1_frequencies[feature] = 0.0

    if total_sepsis_0_entries == 0 and total_sepsis_1_entries == 0 and psv_files:
        print("No entries found to analyze across all processed files, or errors prevented data aggregation.")
        return
    elif not psv_files: # Should have been caught earlier
        return 

    feature_categories = {
        "VITAL SIGNS": VITAL_SIGNS,
        "LABORATORY TESTS": LAB_TESTS,
        "DEMOGRAPHICS & ADMINISTRATIVE": DEMOGRAPHICS 
    }

    print_entry_level_comparison_table(
        group1_name="SepsisLabel=1 Entries", 
        group1_frequencies=sepsis_1_frequencies,
        group1_count=total_sepsis_1_entries,
        group2_name="SepsisLabel=0 Entries", 
        group2_frequencies=sepsis_0_frequencies,
        group2_count=total_sepsis_0_entries,
        features_to_analyze=features_to_analyze,
        feature_categories=feature_categories
    )

if __name__ == '__main__':
    main() 