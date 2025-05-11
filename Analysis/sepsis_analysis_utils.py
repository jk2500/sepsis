import os
import glob
from collections import defaultdict
import concurrent.futures # Added for multithreading

def get_column_headers(file_path):
    """Reads the header row from a PSV file."""
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        return header.split('|')

def check_if_sepsis_case(file_path):
    """
    Checks if a patient file represents a sepsis case.
    A patient is considered a sepsis case if SepsisLabel=1 appears at least once in their file.
    
    Returns:
        bool: True if the patient developed sepsis, False otherwise
    """
    with open(file_path, 'r') as f:
        # Get headers to find SepsisLabel index
        headers = f.readline().strip().split('|')
        try:
            sepsis_index = headers.index('SepsisLabel')
        except ValueError:
            # If SepsisLabel is not in headers, return False
            return False
        
        # Check each line for SepsisLabel=1
        for line in f:
            values = line.strip().split('|')
            if len(values) > sepsis_index and values[sepsis_index] == '1':
                return True
    
    return False

def process_patient_file_for_analysis(file_path, all_headers, columns_to_drop=None):
    """
    Processes a single PSV file to calculate feature frequencies,
    determine sepsis status, and calculate sepsis duration.
    Frequency = (count of non-NaN values) / (total number of data rows).
    
    Args:
        file_path: Path to the PSV file.
        all_headers: List of all column headers from the dataset.
        columns_to_drop: List of column names to exclude from frequency analysis.

    Returns:
        A dictionary containing:
        - 'frequencies': {feature: frequency}
        - 'rows': total number of data rows in the file
        - 'is_sepsis': boolean, True if SepsisLabel=1 appears at least once
        - 'duration': integer, count of rows where SepsisLabel=1
    """
    if columns_to_drop is None:
        columns_to_drop = []

    headers_for_freq_analysis = [h for h in all_headers if h not in columns_to_drop]
    # Need to get original indices from all_headers for these specific headers_for_freq_analysis
    # This ensures we are looking at the correct columns in the file, even if some are dropped for frequency counting.
    indices_for_freq_analysis = [i for i, header in enumerate(all_headers) if header in headers_for_freq_analysis]

    feature_counts = defaultdict(int)
    num_data_rows = 0
    is_sepsis_case = False
    sepsis_duration_hours = 0
    sepsis_label_idx = -1

    try:
        # We need all_headers here to correctly find SepsisLabel, not the filtered ones.
        sepsis_label_idx = all_headers.index('SepsisLabel')
    except ValueError:
        # SepsisLabel column not found in all_headers, cannot determine sepsis status/duration
        # This is not an error for the file, just means we can't track sepsis for it.
        pass 

    with open(file_path, 'r') as f:
        f.readline() # Skip header line in the file itself
        for line in f:
            num_data_rows += 1
            values = line.strip().split('|')

            # Calculate feature frequencies using the pre-calculated indices_for_freq_analysis
            for original_idx in indices_for_freq_analysis:
                header_name = all_headers[original_idx] 
                if original_idx < len(values) and values[original_idx].lower() != 'nan' and values[original_idx].strip() != '':
                    feature_counts[header_name] += 1 # Use the actual header name for the key
            
            # Check for sepsis using sepsis_label_idx derived from all_headers
            if sepsis_label_idx != -1 and len(values) > sepsis_label_idx:
                if values[sepsis_label_idx] == '1':
                    is_sepsis_case = True
                    sepsis_duration_hours += 1
    
    if num_data_rows == 0:
        return {
            'frequencies': {header: 0.0 for header in headers_for_freq_analysis},
            'rows': 0,
            'is_sepsis': False,
            'duration': 0
        }

    # Calculate frequencies only for headers_for_freq_analysis
    feature_frequencies = {
        header_name: (feature_counts[header_name] / num_data_rows) for header_name in headers_for_freq_analysis
    }
    
    return {
        'frequencies': feature_frequencies,
        'rows': num_data_rows,
        'is_sepsis': is_sepsis_case,
        'duration': sepsis_duration_hours
    }

def print_nicely_formatted_results(headers, average_frequencies, files_processed_count, cohort_name="ALL"):
    """
    Prints the results in a nicely formatted manner with categorization.
    
    Args:
        headers: List of column headers to include in the output
        average_frequencies: Dictionary mapping features to their average frequencies
        files_processed_count: Number of files/samples processed for this analysis
        cohort_name: Optional name of the cohort being analyzed (default: "ALL")
    """
    
    # Define categories for different types of features
    vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    lab_tests = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                'Fibrinogen', 'Platelets']
    demographics = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
    
    # Convert frequencies to percentages for better readability
    percentages = {feature: avg_freq * 100 for feature, avg_freq in average_frequencies.items()}
    
    # Define color helpers if terminal supports it
    try:
        # ANSI color codes
        BOLD = '\033[1m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        ENDC = '\033[0m'
        
        def color_by_frequency(freq):
            """Returns color based on frequency value."""
            if freq >= 80:
                return GREEN
            elif freq >= 40:
                return YELLOW
            elif freq > 0:
                return RED
            else:
                return ENDC
    except:
        # Fallback if terminal doesn't support colors
        BOLD = GREEN = YELLOW = RED = BLUE = ENDC = ''
        def color_by_frequency(freq):
            return ''
    
    # Print the header with summary info
    title = f"===== FEATURE RECORDING FREQUENCY ANALYSIS"
    sample_type = "patient files" if cohort_name != "SepsisLabel=0 Entries" and cohort_name != "SepsisLabel=1 Entries" else "entries"
    if cohort_name != "ALL":
        title += f": {cohort_name.upper()} COHORT"
    title += " ====="
    print(f"\n{BOLD}{title}{ENDC}")
    print(f"{BOLD}Based on analysis of {files_processed_count} {sample_type}{ENDC}")
    print("Frequency = percentage of time points where the feature has a value\n")
    
    # Print table header
    header_format = f"{BOLD}{'RANK':<5} {'FEATURE':<25} {'FREQUENCY (%)':<15} {'AVAILABILITY':<15}{ENDC}"
    print(header_format)
    print("-" * 60)
    
    # Helper function to print a category section
    def print_category(category_name, feature_list):
        # Filter feature list to only include features we have data for
        available_features = [f for f in feature_list if f in percentages]
        if not available_features:
            return  # Skip this category if all its features were dropped
            
        # Sort features by percentage (availability) in descending order
        sorted_features = sorted(available_features, key=lambda f: percentages[f], reverse=True)
        
        print(f"\n{BOLD}{BLUE}{category_name}{ENDC}")
        print("-" * 60)
        
        for rank, feature in enumerate(sorted_features, 1):
            percentage = percentages[feature]
            availability = "High" if percentage >= 80 else "Medium" if percentage >= 40 else "Low" if percentage > 0 else "None"
            color = color_by_frequency(percentage)
            print(f"{rank:<5} {feature:<25} {color}{percentage:>5.1f}%{ENDC:<9} {availability:<15}")
    
    # Print features by category
    print_category("VITAL SIGNS", vital_signs)
    print_category("LABORATORY TESTS", lab_tests)
    # Filter demographics for what's actually in `average_frequencies` to avoid errors if some are dropped
    demographics_to_print = [d for d in demographics if d in average_frequencies]
    if demographics_to_print:
        print_category("DEMOGRAPHICS & ADMINISTRATIVE", demographics_to_print)
    
    # Print overall statistics
    high_avail = sum(1 for f in average_frequencies if percentages.get(f,0) >= 80)
    med_avail = sum(1 for f in average_frequencies if 40 <= percentages.get(f,0) < 80)
    low_avail = sum(1 for f in average_frequencies if 0 < percentages.get(f,0) < 40)
    no_avail = sum(1 for f in average_frequencies if percentages.get(f,0) == 0)
    
    summary_title = f"SUMMARY"
    if cohort_name != "ALL":
        summary_title += f" FOR {cohort_name.upper()} COHORT"
    print(f"\n{BOLD}{summary_title}:{ENDC}")
    print(f"High availability features (≥80%): {high_avail}")
    print(f"Medium availability features (40-79%): {med_avail}")
    print(f"Low availability features (<40%): {low_avail}")
    print(f"Unavailable features (0%): {no_avail}")
    
    return percentages

def compare_cohorts(sepsis_frequencies, non_sepsis_frequencies, sepsis_files_count, non_sepsis_files_count):
    """
    Prints a comparison of feature frequencies between sepsis and non-sepsis cohorts.
    Categorizes features and sorts by absolute difference in availability within each category.
    
    Args:
        sepsis_frequencies: Dictionary mapping features to frequencies for sepsis cohort
        non_sepsis_frequencies: Dictionary mapping features to frequencies for non-sepsis cohort
        sepsis_files_count: Number of patient files in the sepsis cohort
        non_sepsis_files_count: Number of patient files in the non-sepsis cohort
    """
    if not sepsis_frequencies or not non_sepsis_frequencies:
        print("Cannot compare cohorts: missing data for one or both cohorts.")
        return
    
    vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    lab_tests = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                'Fibrinogen', 'Platelets']
    demographics = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'] 

    common_features = set(sepsis_frequencies.keys()) & set(non_sepsis_frequencies.keys())
    
    differences = {
        feature: (sepsis_frequencies.get(feature, 0) - non_sepsis_frequencies.get(feature, 0)) * 100
        for feature in common_features
    }
    
    try:
        BOLD = '\033[1m'
        GREEN = '\033[92m' 
        RED = '\033[91m'   
        BLUE = '\033[94m'  
        ENDC = '\033[0m'
    except:
        BOLD = GREEN = RED = BLUE = ENDC = ''
    
    print(f"\n{BOLD}===== COMPARISON: SEPSIS vs NON-SEPSIS COHORTS (Sorted by Absolute Difference within Category) ====={ENDC}")
    print(f"{BOLD}Sepsis Cohort: {sepsis_files_count} patient files | Non-Sepsis Cohort: {non_sepsis_files_count} patient files{ENDC}")

    column_header_format = f"{BOLD}{'FEATURE':<25} {'SEPSIS (%)':<15} {'NON-SEPSIS (%)':<15} {'DIFFERENCE (%)':<15}{ENDC}"

    def print_category_comparison(category_name, feature_list_for_category):
        print(f"\n{BOLD}{BLUE}{category_name}{ENDC}")
        print(column_header_format)
        print("-" * 70)
        
        category_features_in_comparison = [f for f in feature_list_for_category if f in common_features]
        
        # Sort by absolute difference in frequencies (descending)
        sorted_category_features = sorted(category_features_in_comparison, key=lambda f: abs(differences.get(f, 0)), reverse=True)
        
        if not sorted_category_features:
            print("No common features to display for this category.")
            return

        for feature in sorted_category_features:
            sepsis_pct = sepsis_frequencies.get(feature, 0) * 100
            non_sepsis_pct = non_sepsis_frequencies.get(feature, 0) * 100
            diff_val = differences.get(feature, 0)
            color = GREEN if diff_val > 0 else RED if diff_val < 0 else ''
            print(f"{feature:<25} {sepsis_pct:>5.1f}%{'':<9} {non_sepsis_pct:>5.1f}%{'':<9} {color}{diff_val:>+5.1f}%{ENDC:<9}")

    print_category_comparison("VITAL SIGNS", vital_signs)
    print_category_comparison("LABORATORY TESTS", lab_tests)
    
    demographic_features_in_comparison = [f for f in demographics if f in common_features]
    if demographic_features_in_comparison:
        print_category_comparison("DEMOGRAPHICS & ADMINISTRATIVE (if present)", demographic_features_in_comparison)

def find_psv_files(data_directory, limit=None):
    """
    Finds and returns PSV files in the specified directory.
    
    Args:
        data_directory: Directory to search for PSV files
        limit: Optional limit on the number of files to return
    
    Returns:
        List of file paths
    """
    # Construct an absolute path if data_directory is relative
    if not os.path.isabs(data_directory):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(script_dir)
        data_directory = os.path.join(workspace_root, data_directory)

    psv_files = sorted(glob.glob(os.path.join(data_directory, 'p*.psv')))
    
    if limit and len(psv_files) > limit:
        return psv_files[:limit]
    
    return psv_files

def analyze_cohort(files, all_headers, columns_to_drop, cohort_name="ALL"):
    """
    Analyzes feature frequencies and sepsis statistics for a specific cohort of patients.
    Uses multithreading to process files in parallel.
    Each file is read only once.
    
    Args:
        files: List of file paths for patients in this cohort
        all_headers: List of all column headers (obtained from one example file)
        columns_to_drop: List of column names to exclude from frequency analysis
        cohort_name: Name of the cohort for display purposes
    
    Returns:
        A dictionary containing:
        - 'avg_freqs': {feature: average_frequency}
        - 'files_count': number of files successfully processed with data
        - 'total_sepsis_cases': count of sepsis cases in the cohort
        - 'total_sepsis_duration': sum of sepsis durations for all sepsis cases
        - 'sepsis_duration_contributors': count of sepsis cases with duration > 0
    """
    print(f"\nAnalyzing {len(files)} files in the {cohort_name} cohort using multiple threads...")
    
    headers_for_freq_reporting = [h for h in all_headers if h not in columns_to_drop]
    
    sum_of_frequencies = defaultdict(float)
    files_processed_with_data_count = 0
    total_sepsis_cases_in_cohort = 0
    total_sepsis_duration_for_cohort = 0
    sepsis_patients_contributing_to_duration = 0
    processed_count_for_progress = 0 # For progress reporting

    # Use a ThreadPoolExecutor to process files in parallel
    # os.cpu_count() can be a good default for I/O bound tasks, adjust if needed
    # If os.cpu_count() returns None, default to a reasonable number like 4. Consider context where os might not be fully available.
    num_workers = os.cpu_count() if os.cpu_count() is not None else 4 
    if not files: # handle empty list of files
        print(f"No files provided to analyze for the {cohort_name} cohort.")
        # Fall through to return empty/zeroed results, which is current behavior for no data.
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {executor.submit(process_patient_file_for_analysis, file, all_headers, columns_to_drop): file for file in files}
            
            total_files_to_process = len(files)
            print(f"Submitted {total_files_to_process} files for processing to {num_workers} worker threads for cohort '{cohort_name}'.")

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                processed_count_for_progress += 1
                try:
                    analysis_results = future.result() # Get the result from the future
                    
                    if analysis_results['rows'] > 0:
                        files_processed_with_data_count += 1
                        for feature, freq in analysis_results['frequencies'].items():
                            sum_of_frequencies[feature] += freq
                        
                        if analysis_results['is_sepsis']:
                            total_sepsis_cases_in_cohort += 1
                            if analysis_results['duration'] > 0:
                                total_sepsis_duration_for_cohort += analysis_results['duration']
                                sepsis_patients_contributing_to_duration += 1
                    
                    # Progress reporting (consider adjusting frequency for very large datasets)
                    if processed_count_for_progress % (total_files_to_process // 10 if total_files_to_process >=10 else 1) == 0 or processed_count_for_progress == total_files_to_process:
                        print(f"Cohort '{cohort_name}': Processed {processed_count_for_progress}/{total_files_to_process} files...")

                except FileNotFoundError:
                    print(f"Warning: File not found {file_path} (skipped). Cohort: {cohort_name}")
                except Exception as exc:
                    print(f"Error processing file {os.path.basename(file_path)} for cohort {cohort_name}: {exc} (skipped)")

    if files_processed_with_data_count == 0 and files: # Check if files list was not empty initially
        print(f"No files with data were successfully processed for the {cohort_name} cohort.")
        # Return zeroed results, consistent with original behavior
        return {
            'avg_freqs': {header: 0.0 for header in headers_for_freq_reporting},
            'files_count': 0,
            'total_sepsis_cases': 0,
            'total_sepsis_duration': 0,
            'sepsis_duration_contributors': 0
        }
    elif not files: # If initial files list was empty
         return {
            'avg_freqs': {header: 0.0 for header in headers_for_freq_reporting},
            'files_count': 0,
            'total_sepsis_cases': 0,
            'total_sepsis_duration': 0,
            'sepsis_duration_contributors': 0
        }


    average_frequencies = {}
    if files_processed_with_data_count > 0: 
        average_frequencies = {
            feature: (sum_of_frequencies[feature] / files_processed_with_data_count)
            for feature in headers_for_freq_reporting 
        }
    else: 
        average_frequencies = {header: 0.0 for header in headers_for_freq_reporting}

    print_nicely_formatted_results(headers_for_freq_reporting, average_frequencies, files_processed_with_data_count, cohort_name)

    return {
        'avg_freqs': average_frequencies,
        'files_count': files_processed_with_data_count,
        'total_sepsis_cases': total_sepsis_cases_in_cohort,
        'total_sepsis_duration': total_sepsis_duration_for_cohort,
        'sepsis_duration_contributors': sepsis_patients_contributing_to_duration
    }

def print_entry_level_comparison_table(group1_name, group1_frequencies, group1_count, group2_name, group2_frequencies, group2_count, features_to_analyze, feature_categories):
    """
    Prints a categorized comparison of feature frequencies between two groups of entries.
    Sorts by absolute difference in availability within each category.

    Args:
        group1_name (str): Name of the first group (e.g., "SepsisLabel=1 Entries")
        group1_frequencies (dict): {feature: frequency} for the first group
        group1_count (int): Number of entries in the first group
        group2_name (str): Name of the second group (e.g., "SepsisLabel=0 Entries")
        group2_frequencies (dict): {feature: frequency} for the second group
        group2_count (int): Number of entries in the second group
        features_to_analyze (list): List of features that were included in the frequency calculation.
        feature_categories (dict): {"CATEGORY_NAME": [feature_list]} for display grouping.
    """
    if not group1_frequencies and not group2_frequencies:
        print("Cannot compare entry groups: no frequency data provided for either group.")
        return

    differences = {}
    for feature in features_to_analyze:
        freq1 = group1_frequencies.get(feature, 0)
        freq2 = group2_frequencies.get(feature, 0)
        differences[feature] = (freq1 - freq2) * 100
    
    try:
        BOLD = '\033[1m'
        GREEN = '\033[92m' 
        RED = '\033[91m'   
        BLUE = '\033[94m'  
        ENDC = '\033[0m'
    except:
        BOLD = GREEN = RED = BLUE = ENDC = ''
    
    print(f"\n{BOLD}===== ENTRY-LEVEL COMPARISON: {group1_name.upper()} vs {group2_name.upper()} ====={ENDC}")
    print(f"{BOLD}{group1_name}: {group1_count} entries | {group2_name}: {group2_count} entries{ENDC}")
    print(f"(Sorted by Absolute Difference within Category)")
    
    g1_short_name = group1_name.split(' ')[0].upper()
    g2_short_name = group2_name.split(' ')[0].upper()
    column_header_format = f"{BOLD}{'FEATURE':<25} {g1_short_name:<15} {g2_short_name:<15} {'DIFFERENCE (%)':<15}{ENDC}"
    table_width = 25 + 15 + 15 + 15 + (len(ENDC) * 3) # Basic width, adjust as needed

    for category_name, feature_list_for_category in feature_categories.items():
        category_features_in_analysis = [f for f in feature_list_for_category if f in features_to_analyze]
        
        if not category_features_in_analysis:
            continue

        print(f"\n{BOLD}{BLUE}{category_name}{ENDC}")
        print(column_header_format)
        print("-" * table_width)
        
        # Sort by absolute difference in frequencies (descending)
        sorted_category_features = sorted(category_features_in_analysis, key=lambda f: abs(differences.get(f, 0)), reverse=True)
        
        if not sorted_category_features:
            print("No features to display for this category.")
            continue

        for feature in sorted_category_features:
            freq1_pct = group1_frequencies.get(feature, 0) * 100
            freq2_pct = group2_frequencies.get(feature, 0) * 100
            diff_val = differences.get(feature, 0)
            color = GREEN if diff_val > 0 else RED if diff_val < 0 else ''
            print(f"{feature:<25} {freq1_pct:>5.1f}%{'':<{15-6}} {freq2_pct:>5.1f}%{'':<{15-6}} {color}{diff_val:>+5.1f}%{ENDC:<9}") 

def get_sepsis_duration_for_patient(file_path):
    """
    Counts the number of hours (rows) a patient has SepsisLabel=1.

    Args:
        file_path (str): Path to the patient's PSV file.

    Returns:
        int: Number of hours with SepsisLabel=1. Returns 0 if SepsisLabel column is missing or no sepsis entries.
    """
    hours_in_sepsis = 0
    try:
        with open(file_path, 'r') as f:
            headers = f.readline().strip().split('|')
            try:
                sepsis_label_idx = headers.index('SepsisLabel')
            except ValueError:
                # SepsisLabel column not found in this file
                return 0 
            
            for line in f:
                values = line.strip().split('|')
                if len(values) > sepsis_label_idx and values[sepsis_label_idx] == '1':
                    hours_in_sepsis += 1
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path} when calculating sepsis duration.")
        return 0 # Or handle as an error
    except Exception as e:
        print(f"Warning: Could not read or process {file_path} for sepsis duration: {e}")
        return 0 # Or handle as an error
    return hours_in_sepsis 

def display_average_sepsis_duration(sepsis_patient_files, cohort_name="Sepsis Cases"):
    """
    Calculates and prints the average duration of sepsis for a given cohort of patient files.

    Args:
        sepsis_patient_files (list): List of file paths for patients who had sepsis.
        cohort_name (str): Name of the cohort for display purposes.
    """
    if not sepsis_patient_files:
        print(f"\nNo patient files provided for {cohort_name} to calculate average sepsis duration.")
        return

    total_sepsis_hours = 0
    patients_with_sepsis_duration_data = 0
    
    # Attempt to use ANSI color codes for nicer output
    try:
        BOLD = '\033[1m'
        ENDC = '\033[0m'
    except:
        BOLD = ENDC = '' # Fallback if not supported

    print(f"\n{BOLD}Calculating average sepsis duration for {len(sepsis_patient_files)} files in the {cohort_name} cohort...{ENDC}")

    for i, file_path in enumerate(sepsis_patient_files):
        # Show progress for larger lists
        if (i+1) % 10 == 0 or i == 0 or i == len(sepsis_patient_files) - 1:
            print(f"Processing file {i+1}/{len(sepsis_patient_files)} for duration: {os.path.basename(file_path)}")
        
        duration = get_sepsis_duration_for_patient(file_path)
        
        # We are iterating over files pre-identified as sepsis cases.
        # So, duration should ideally be > 0.
        if duration > 0:
            total_sepsis_hours += duration
            patients_with_sepsis_duration_data +=1
        else:
            # This indicates a file was deemed a sepsis case by one logic (e.g., check_if_sepsis_case)
            # but get_sepsis_duration_for_patient found 0 hours of SepsisLabel=1.
            # This could be due to data nuances or if SepsisLabel=1 appeared only on the header/malformed line for that check.
            print(f"Note: Patient {os.path.basename(file_path)} from '{cohort_name}' cohort showed 0 hours with SepsisLabel=1 via get_sepsis_duration_for_patient.")

    print(f"\n{BOLD}===== AVERAGE SEPSIS DURATION ANALYSIS: {cohort_name.upper()} ====={ENDC}")
    if patients_with_sepsis_duration_data > 0:
        average_duration = total_sepsis_hours / patients_with_sepsis_duration_data
        print(f"Cohort Analyzed: {cohort_name}")
        print(f"Total patient files initially identified as sepsis cases for this analysis: {len(sepsis_patient_files)}")
        print(f"Number of these patients for whom SepsisLabel=1 entries were found and duration calculated: {patients_with_sepsis_duration_data}")
        print(f"Total hours with SepsisLabel=1 across these {patients_with_sepsis_duration_data} patients: {total_sepsis_hours} hours")
        print(f"Average duration of SepsisLabel=1 per patient (with duration data): {average_duration:.2f} hours{ENDC}")
    else:
        print(f"No SepsisLabel=1 entries found (or durations calculated as > 0) across the provided {len(sepsis_patient_files)} patient files in {cohort_name}.")
        print(f"Cannot calculate average sepsis duration.{ENDC}") 