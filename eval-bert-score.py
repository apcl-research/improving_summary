import sys
import pickle
import numpy as np
import os
import argparse
from bert_score import score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define paths
datapath = '.'
outpath = '.'


def fil(com):
    """Filter function to remove special characters from tokens."""
    return [w for w in com if '<' not in w]


def prep_dataset2(predFile, refFile):
    """Reads prediction and reference files and structures the data."""
    with open(refFile, 'r', encoding="utf-8") as ref, open(predFile, 'r', encoding="utf-8") as predicts:
        refcoms, predcoms, fidlist = [], [], []
        preds = {}

        # Read predictions
        for line in predicts:
            try:
                fid, pred = line.strip().split(';\t')
                fid = int(fid)
                pred = fil(pred.split())
                preds[fid] = pred
            except ValueError:
                print(f"Error reading prediction line: {repr(line)}")

        # Read references and match with predictions
        for line in ref:
            try:
                fid, com = line.strip().split(';\t')
                fid = int(fid)
                com = fil(com.split())

                if len(com) < 1:
                    continue

                if fid in preds:
                    pred_text = " ".join(preds[fid])
                    ref_text = " ".join(com)

                    # Skip if reference and prediction are exactly the same
                    if pred_text == ref_text:
                        print(f"Skipping identical summary for ID {fid}")
                        continue

                    predcoms.append(preds[fid])
                    refcoms.append(com)
                    fidlist.append(fid)
                else:
                    print(f"Warning: No prediction found for reference ID {fid}")

            except ValueError:
                print(f"Error reading reference line: {repr(line)}")

    return fidlist, refcoms, predcoms


def calculate_bert_score(fidlist, reflist, predlist):
    """Calculates BERTScore for all reference-prediction pairs."""
    refs = [" ".join(ref) for ref in reflist]
    preds = [" ".join(pred) for pred in predlist]

    # Compute BERTScore
    p, r, f1 = score(preds, refs, lang='en', rescale_with_baseline=True)

    # Store individual scores
    p_bert, r_bert, f1_bert = {}, {}, {}
    for fid, pscore, rscore, f1score in zip(fidlist, p.numpy(), r.numpy(), f1.numpy()):
        p_bert[fid] = pscore
        r_bert[fid] = rscore
        f1_bert[fid] = f1score

    # Save results
    pickle.dump(p_bert, open(outpath + '/pbert.pkl', 'wb'))
    pickle.dump(r_bert, open(outpath + '/rbert.pkl', 'wb'))
    pickle.dump(f1_bert, open(outpath + '/f1bert.pkl', 'wb'))

    # Compute average scores
    avg_p = np.mean(list(p_bert.values())) * 100
    avg_r = np.mean(list(r_bert.values())) * 100
    avg_f1 = np.mean(list(f1_bert.values())) * 100

    # Print results
    print(f"\nBERTScore for {len(predlist)} functions:")
    print(f"Precision: {avg_p:.2f}")
    print(f"Recall: {avg_r:.2f}")
    print(f"F1-score: {avg_f1:.2f}")

    return p_bert, r_bert, f1_bert


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate BERTScore for reference and prediction summaries.')
    parser.add_argument('predFile', type=str, help="Path to the prediction file.")
    parser.add_argument('refFile', type=str, help="Path to the reference file.")

    args = parser.parse_args()
    predFile = args.predFile
    refFile = args.refFile

    # Prepare dataset
    fidlist, refcoms, predcoms = prep_dataset2(predFile, refFile)

    # Compute BERTScore
    calculate_bert_score(fidlist, refcoms, predcoms)
