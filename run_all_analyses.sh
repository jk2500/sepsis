#!/bin/bash

# Create an output directory if it doesn't exist
OUTPUT_DIR="analysis_outputs"
mkdir -p "$OUTPUT_DIR"

echo "Running Analysis/average_frequency_calculator.py..."
python Analysis/average_frequency_calculator.py > "$OUTPUT_DIR/average_frequency_calculator_output.txt"
echo "Output saved to $OUTPUT_DIR/average_frequency_calculator_output.txt"
echo "------------------------------------"

echo "Running Analysis/cohort_frequency_analysis.py..."
python Analysis/cohort_frequency_analysis.py > "$OUTPUT_DIR/cohort_frequency_analysis_output.txt"
echo "Output saved to $OUTPUT_DIR/cohort_frequency_analysis_output.txt"
echo "------------------------------------"

echo "Running Analysis/entry_level_sepsis_comparison.py..."
python Analysis/entry_level_sepsis_comparison.py > "$OUTPUT_DIR/entry_level_sepsis_comparison_output.txt"
echo "Output saved to $OUTPUT_DIR/entry_level_sepsis_comparison_output.txt"
echo "------------------------------------"

echo "Running Analysis/feature_mutual_information_pairwise.py..."
python Analysis/feature_mutual_information_pairwise.py > "$OUTPUT_DIR/feature_mutual_information_pairwise_output.txt"
echo "Output saved to $OUTPUT_DIR/feature_mutual_information_pairwise_output.txt"
echo "------------------------------------"

echo "All analyses complete. Outputs are in the '$OUTPUT_DIR' directory." 