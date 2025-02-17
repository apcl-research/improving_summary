import sys
import numpy as np
import os
import argparse
import collections
from scipy.stats import ttest_rel
from bert_score import score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fil(com):
    """Filters out unwanted tokens (e.g., those containing '<')."""
    return [w for w in com if '<' not in w]


def prep_dataset_bert(comsfile, inputA_file, inputB_file, delim, diffonly):
    """
    Prepares the dataset for BERTScore evaluation.
    Reads predictions from Model A and Model B (from .txt files) and
    the reference comments from the coms file (.test file). Applies similar
    filtering rules as in your USE code.
    """
    # Prepare predictions for Model A
    predsA = dict()
    with open(inputA_file, 'r', encoding='utf-8') as predictsA:
        for line in predictsA:
            try:
                # Expecting format: fid ;\t prediction_text
                fid, pred = line.strip().split(';\t')
                fid = int(fid)
                pred = pred.split()
                pred = fil(pred)
                predsA[fid] = pred
            except ValueError:
                print(f"Error reading Model A prediction line: {repr(line)}")
                continue

    # Prepare predictions for Model B
    predsB = dict()
    with open(inputB_file, 'r', encoding='utf-8') as predictsB:
        for line in predictsB:
            try:
                fid, pred = line.strip().split(';\t')
                fid = int(fid)
                pred = pred.split()
                pred = fil(pred)
                predsB[fid] = pred
            except ValueError:
                print(f"Error reading Model B prediction line: {repr(line)}")
                continue

    refs = []
    newpredsA = []
    newpredsB = []
    # In case you want to evaluate cases where the two models produce identical predictions
    samesPreds = []
    samesRefs = []

    with open(comsfile, 'r', encoding='utf-8') as targets:
        for line in targets:
            try:
                # The reference file is assumed to have the format: fid <delim> reference_text
                fid, com = line.strip().split(delim)
                fid = int(fid)
                com = com.split()
                com = fil(com)
            except ValueError:
                print(f"Error reading reference line: {repr(line)}")
                continue

            if len(com) < 1:
                continue

            if fid in predsA and fid in predsB:
                # (Optional) Skip if Model B's prediction exactly equals the reference.
                if " ".join(predsB[fid]).strip() == " ".join(com).strip():
                    continue

                # If diffonly is set, then we separate out cases where the predictions are identical.
                if diffonly and predsA[fid] == predsB[fid]:
                    samesPreds.append(predsA[fid])
                    samesRefs.append(com)
                    continue

                newpredsA.append(predsA[fid])
                newpredsB.append(predsB[fid])
                refs.append(com)

        # Compute BERTScore F1 for Model A and Model B
        scoresA = calculate_bert_f1(refs, newpredsA)
        avgA = np.mean(scoresA) * 100
        print(f"Model A average BERT F1: {avgA:.2f}")

        scoresB = calculate_bert_f1(refs, newpredsB)
        avgB = np.mean(scoresB) * 100
        print(f"Model B average BERT F1: {avgB:.2f}")

        # Perform a paired t-test.
        # (Here the alternative hypothesis is that Model A's scores are greater than Model B's.)
        ttest = ttest_rel(scoresA, scoresB, alternative='greater')
        print("\nFinal t-test result (Model A > Model B):", ttest)

        # (Optional) Evaluate the subset where predictions are identical across models.
        if diffonly and len(samesRefs) > 0:
            scoresSame = calculate_bert_f1(samesRefs, samesPreds)
            avgSame = np.mean(scoresSame) * 100
            print(f"Identical predictions group average BERT F1: {avgSame:.2f}")

    return ttest


def calculate_bert_f1(refs, preds):
    """
    Computes BERTScore F1 scores for a list of prediction/reference pairs.
    The function converts token lists into strings and calls bert_score.score().
    """
    # Convert list-of-tokens to strings
    refs_text = [" ".join(ref) for ref in refs]
    preds_text = [" ".join(pred) for pred in preds]

    # Compute BERTScore; we only use the F1 scores here.
    # (You can also retrieve precision and recall if needed.)
    _, _, f1 = score(preds_text, refs_text, lang='en', rescale_with_baseline=True)
    return f1.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate statistical significance (paired t-test) using BERTScore for two prediction files.'
    )
    parser.add_argument('inputA', type=str, help="First prediction file (Model A, .txt)")
    parser.add_argument('inputB', type=str, help="Second prediction file (Model B, .txt)")
    parser.add_argument('--coms-filename', dest='comsfilename', type=str, default='coms.test',
                        help="Reference file (coms file, .test)")
    # Use the appropriate delimiter for your reference file; default here is '<SEP>'
    parser.add_argument('--delim', dest='delim', type=str, default='<SEP>',
                        help="Delimiter used in the reference file")
    # When diffonly is True, we exclude cases where the two models’ predictions are identical.
    parser.add_argument('--not-diffonly', dest='diffonly', action='store_false', default=True,
                        help="Include predictions that are identical across models")
    args = parser.parse_args()

    # Prepare dataset (the coms file is the reference)
    ttest = prep_dataset_bert(
        args.comsfilename, args.inputA, args.inputB, args.delim, args.diffonly
    )

