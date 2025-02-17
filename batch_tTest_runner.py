import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from scipy.stats import ttest_rel
from use_score_v_diff import prep_dataset  # Import your existing function

# Set environment variables for TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Batch T-Test Runner for Model Predictions")
parser.add_argument("--not-diffonly", dest="diffonly", action="store_false", default=True,
                    help="Include all predictions in evaluation, not just differing ones.")

args = parser.parse_args()

# Define input directories
model_a_dir = "jam_cgpt_predictions/Condensing_preds/"  # Change this to your Model A directory
model_b_dir = "../methods/Condensing/"  # Change this to your Baseline's directory
coms_file = "data/Condensing/Human_.test"  # Change this to your reference file
batchsize = 50000  # Adjust batch size based on memory

# Get all prediction files in the directories
model_a_files = sorted(glob.glob(os.path.join(model_a_dir, "*.txt")))
model_b_files = sorted(glob.glob(os.path.join(model_b_dir, "*.txt")))

# Output file to store results
output_file = "ttest_results_condensing_notdiff.csv"

# Open the results file
with open(output_file, "w") as f:
    f.write("Prediction,Baseline,T-statistic,P-value\n")
    # Loop through each pair of prediction files
    for file_b in model_b_files:
        for file_a in model_a_files:
            print(f"\nProcessing: {file_a} vs {file_b}")

            # Compute paired t-test
            ttest_result = prep_dataset(coms_file, file_a, file_b, "<SEP>", batchsize, diffonly=args.diffonly)

            # Extract t-test values
            t_stat, p_value = ttest_result.statistic, ttest_result.pvalue

            # Write results to file
            f.write(f"{os.path.basename(file_a)},{os.path.basename(file_b)},{t_stat},{p_value}\n")

            # Print results
            print(f"T-test Statistic: {t_stat}, P-value: {p_value}")

print("\n✅ All comparisons complete. Results saved.'")
