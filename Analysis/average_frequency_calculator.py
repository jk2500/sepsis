import os
from sepsis_analysis_utils import (
    get_column_headers,
    find_psv_files,
    analyze_cohort
    # Removed: check_if_sepsis_case, get_sepsis_duration_for_patient
    # print_nicely_formatted_results is called within analyze_cohort
)

def main():
    data_directory = './physionet.org/files/challenge-2019/1.0.0/training/training_setB/'
    
    # Define columns to drop from the frequency analysis part
    # SepsisLabel is dropped for frequency reporting but still used internally by analyze_cohort for sepsis stats
    columns_to_drop_freq_analysis = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'Age', 'Gender', 'SepsisLabel']
    
    # Find PSV files (limit for faster processing)
    # Set limit to None for all files for a full analysis
    psv_files = find_psv_files(data_directory, limit=None) # User had limit=None, using 100 for consistency with other files for now
    
    if not psv_files:
        print(f"No PSV files found in {data_directory}")
        return

    example_file_path = psv_files[0]
    all_headers = get_column_headers(example_file_path)
    if not all_headers:
        print(f"Could not read headers from {example_file_path}")
        return
        
    # --- Perform Combined Analysis (Feature Frequencies and Sepsis Stats) --- 
    # analyze_cohort will print its own feature frequency report.
    # It returns a dictionary with all aggregated stats.
    print(f"\nProcessing {len(psv_files)} files for combined analysis...")
    if columns_to_drop_freq_analysis:
      print(f"(The following columns will be dropped from the FEATURE FREQUENCY part of the analysis: {', '.join(columns_to_drop_freq_analysis)})")
    
    # The cohort_name for analyze_cohort becomes the title for the frequency report part.
    analysis_results = analyze_cohort(
        psv_files, 
        all_headers, 
        columns_to_drop_freq_analysis, 
        cohort_name="ALL PROCESSED FILES (SINGLE PASS)"
    )

    # --- Extract and Display Average Sepsis Duration from analysis_results --- 
    files_processed_count = analysis_results.get('files_count', 0) # Number of files with data rows
    total_sepsis_cases = analysis_results.get('total_sepsis_cases', 0)
    total_sepsis_duration_hours = analysis_results.get('total_sepsis_duration', 0)
    sepsis_duration_contributors = analysis_results.get('sepsis_duration_contributors', 0) # Patients who had SepsisLabel=1 for >0 hours

    # For colored/bold output, consistent with other utility prints
    try:
        BOLD = '\033[1m'
        ENDC = '\033[0m'
    except: # Should not fail with direct assignment but good practice
        BOLD = ENDC = ''

    print(f"\n\n{BOLD}===== AVERAGE SEPSIS DURATION (from single pass analysis) ====={ENDC}")
    print(f"{BOLD}Based on the same {files_processed_count} patient files processed for feature frequencies.{ENDC}")

    if total_sepsis_cases > 0:
        if sepsis_duration_contributors > 0:
            average_sepsis_duration = total_sepsis_duration_hours / sepsis_duration_contributors
            print(f"Total sepsis cases identified: {total_sepsis_cases}")
            print(f"Number of these sepsis patients with SepsisLabel=1 entries (contributing to duration): {sepsis_duration_contributors}")
            print(f"Total hours with SepsisLabel=1 across these {sepsis_duration_contributors} patients: {total_sepsis_duration_hours} hours")
            print(f"{BOLD}Average SepsisLabel=1 duration for patients with sepsis entries: {average_sepsis_duration:.2f} hours{ENDC}")
        else:
            print(f"Total sepsis cases identified: {total_sepsis_cases}")
            print(f"However, no SepsisLabel=1 entries (duration > 0 hours) were found for these sepsis cases.")
            print(f"Cannot calculate average sepsis duration based on actual SepsisLabel=1 entries.")
    elif files_processed_count > 0:
        print("\nNo sepsis cases were identified in the processed files.")
    else:
        print("\nNo files were processed or no data found in files.")

if __name__ == '__main__':
    main() 