import subprocess
import re
import sys


def get_bert_score(predfile, reffile):
    """Runs get-bert-score.py and extracts BERTScore (Precision, Recall, F1)."""
    command = ["python3", "eval-bert-score.py", predfile, reffile]
    result = subprocess.run(command, capture_output=True, text=True)

    # Extract precision, recall, and F1-score using regex
    match = re.search(
        r'Precision: ([\d\.]+)\nRecall: ([\d\.]+)\nF1-score: ([\d\.]+)', result.stdout)

    if match:
        precision = float(match.group(1))
        recall = float(match.group(2))
        f1_score = float(match.group(3))
        return precision, recall, f1_score
    else:
        print(f"Error processing {predfile}. Output:\n{result.stdout}")
        return None, None, None


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 script.py <ref_file> <predfile1> <predfile2> ...")
        sys.exit(1)

    ref_file = sys.argv[1]
    pred_files = sys.argv[2:]

    bert_scores = []

    for pred_file in pred_files:
        precision, recall, f1_score = get_bert_score(pred_file, ref_file)

        if precision is not None and recall is not None and f1_score is not None:
            bert_scores.append((precision, recall, f1_score))
            print(f"{pred_file}: Precision = {precision}, Recall = {recall}, F1 = {f1_score}")

    if bert_scores:
        avg_precision = sum(score[0] for score in bert_scores) / len(bert_scores)
        avg_recall = sum(score[1] for score in bert_scores) / len(bert_scores)
        avg_f1 = sum(score[2] for score in bert_scores) / len(bert_scores)

        print("\nAverage BERTScore across files:")
        print(f"Precision: {avg_precision:.2f}")
        print(f"Recall: {avg_recall:.2f}")
        print(f"F1-score: {avg_f1:.2f}")


if __name__ == "__main__":
    main()
