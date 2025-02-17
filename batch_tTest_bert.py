#!/usr/bin/env python3
"""
run_multiple_ttests.py

This script iterates over prediction files (Model A) in a specified directory,
runs the t-test evaluation by invoking `bert_tTest.py` (with Model B being constant),
parses the printed t-test result, and saves the outcome for each file in a CSV.
"""

import subprocess
import glob
import os
import re
import csv


def run_ttest_for_file(pred_file, baseline_file, coms_file):
    """
    Runs the t-test evaluation for a given prediction file using bert_tTest.py.

    :param pred_file: Path to Model A predictions (e.g., "jam_cgpt_predictions/Unambiguous_preds/...")
    :param baseline_file: Path to Model B predictions (e.g., "../methods/Unambiguous/SU1.txt")
    :param coms_file: Path to the reference file (e.g., "data/Unambigious/human.test")
    :return: The standard output from the command as a string, or None if an error occurs.
    """
    # Build the command:
    #   python3 bert_tTest.py <pred_file> <baseline_file> --coms-filename <coms_file> --not-diffonly
    cmd = [
        "python3",
        "bert_tTest.py",
        pred_file,
        baseline_file,
        "--coms-filename", coms_file,
        "--not-diffonly"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {pred_file}:\n{e}")
        return None


def parse_ttest_output(output):
    """
    Extracts the t-test statistic and p-value from the output string.

    The bert_tTest.py script prints a line like:
      Final t-test result (Model A > Model B): TtestResult(statistic=0.1234, pvalue=0.0567)

    :param output: The complete output string from bert_tTest.py
    :return: A tuple (t_stat, p_val) as strings, or (None, None) if not found.
    """
    # Regular expression to capture the numeric values for statistic and pvalue.
    pattern = r"Final t-test result.*TtestResult\(statistic=([\d\.\-eE]+), pvalue=([\d\.\-eE]+)\)"
    match = re.search(pattern, output)
    if match:
        t_stat = match.group(1)
        p_val = match.group(2)
        return t_stat, p_val
    else:
        return None, None


def main():
    # Configuration: update these paths as needed.
    predictions_dir = "jam_cgpt_predictions/Unambiguous_preds"
    baseline_file_dir = "../methods/Unambiguous/SU1.txt"
    coms_file = "data/Unambigious/human.test"
    output_csv = "ttest_results_unambiguous_bert.csv"

    # Find all prediction files (assumes files are named like "predict_Unambiguous_*.txt")
    pattern = os.path.join(predictions_dir, "predict_Unambiguous_E5_*.txt")
    pred_files = glob.glob(pattern)
    baseline_files = glob.glob(os.path.join(baseline_file_dir, "*.txt"))
    if not pred_files:
        print(f"No prediction files found in {predictions_dir} with pattern {pattern}")
        return

    results = []
    for baseline_file in baseline_files:
        for pred_file in pred_files:
            print(f"Running t-test for: {pred_file}")
            output = run_ttest_for_file(pred_file, baseline_file, coms_file)
            if output is None:
                print(f"Skipping {pred_file} due to error.")
                continue

            t_stat, p_val = parse_ttest_output(output)
            if t_stat is None or p_val is None:
                print(f"Could not parse t-test result for {pred_file}. Full output:")
                print(output)

            results.append({
                "prediction_file": pred_file,
                "t_statistic": t_stat if t_stat is not None else "N/A",
                "p_value": p_val if p_val is not None else "N/A",
                "full_output": output.strip()
            })
            print("\nDone\n")

    # Save results to CSV.
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["prediction_file", "t_statistic", "p_value", "full_output"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nAll results have been saved to {output_csv}")


if __name__ == "__main__":
    main()
