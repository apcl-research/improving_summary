import sys
import numpy as np
import os
import argparse
import collections
import tensorflow as tf
from scipy.stats import ttest_rel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fil(com):
    """Filters out unwanted tokens (e.g., containing '<')."""
    return [w for w in com if '<' not in w]


def use(reflist, predlist, batchsize):
    """Computes cosine similarity scores using Google's Universal Sentence Encoder (USE)."""
    import tensorflow_hub as tfhub

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = tfhub.load(module_url)

    refs = []
    preds = []

    for ref, pred in zip(reflist, predlist):
        ref_text = ' '.join(ref).strip()
        pred_text = ' '.join(pred).strip()

        if pred_text == '':
            pred_text = ' <s> '  # Default placeholder for empty predictions

        refs.append(ref_text)
        preds.append(pred_text)

    if not refs:
        return [], 0.0, "No mismatched predictions for evaluation."


    # Compute similarity scores
    scores = []
    for i in range(0, len(refs), batchsize):
        ref_emb = model(refs[i:i + batchsize])
        pred_emb = model(preds[i:i + batchsize])
        csm = cosine_similarity_score(ref_emb, pred_emb)
        csd = csm.diagonal()
        scores.extend(csd.tolist())

    avg_css = np.average(scores) if scores else 0.0

    corpuse = (round(avg_css * 100, 2))
    ret = ('for %s functions\n' % (len(predlist)))
    ret += 'Cosine similarity score with universal sentence encoder embedding is %s\n' % corpuse
    return scores, corpuse, ret


def cosine_similarity_score(x, y):
    """Computes cosine similarity between two embedding sets."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(x, y)


def prep_dataset(comsfile, inputA_file, inputB_file, delim, batchsize, diffonly):
    """Prepares dataset, computes similarity scores using USE."""
    # prep('preparing predictions list A... ')
    predsA = dict()
    predictsA = open(inputA_file, 'r')
    for c, line in enumerate(predictsA):
        try:
            split_line = line.split(';\t')
            (fid, pred) = split_line[0], split_line[-1]
            fid = int(fid)
            pred = pred.split()
            pred = fil(pred)
            predsA[fid] = pred
        except:
            continue

    # prep('preparing predictions list B... ')
    predsB = dict()
    predictsB = open(inputB_file, 'r')
    for c, line in enumerate(predictsB):
        try:
            split_line = line.split(';\t')
            (fid, pred) = split_line[0], split_line[-1]
            fid = int(fid)
            pred = pred.split()
            pred = fil(pred)
            predsB[fid] = pred
        except:
            continue

    refs = list()
    newpredsA = list()
    newpredsB = list()
    samesPreds = list()
    samesRefs = list()

    # targets = open('%s/output/coms.test' % (dataprep), 'r')
    with open(comsfile, 'r') as targets:
        for line in targets:
            try:
                fid, com = line.split(delim)
                fid = int(fid)
                com = fil(com.split())
            except ValueError:
                continue  # Skip malformed lines

            if len(com) < 1:
                continue  # Skip empty references

            if fid in predsA and fid in predsB:
                if predsB[fid] == com:
                    continue  # Skip if Model A's prediction is identical to reference

                if diffonly and predsA[fid] == predsB[fid]:
                    samesPreds.append(predsA[fid])
                    samesRefs.append(com)
                    continue

                newpredsA.append(predsA[fid])
                newpredsB.append(predsB[fid])
                refs.append(com)

    scoresA, SA, ret = use(refs, newpredsA, batchsize)
    print(ret)
    print()

    scoresB, SB, ret = use(refs, newpredsB, batchsize)
    print(ret)
    print()

    if diffonly:
        scoresS, SAMESSCORE, ret = use(samesRefs, samesPreds, batchsize)
        print(ret)
        print()

    ttest = ttest_rel(scoresA, scoresB, alternative='greater')
    return ttest



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate similarity and statistical difference between predictions.')

    parser.add_argument('inputA', type=str, help="First prediction file (Model A)")
    parser.add_argument('inputB', type=str, help="Second prediction file (Model B)")
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/javastmt')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--not-diffonly', dest='diffonly', action='store_false', default=True)
    parser.add_argument('--shuffles', type=int, default=100)
    parser.add_argument('--delim', dest='delim', type=str, default='<SEP>')
    parser.add_argument('--coms-filename', dest='comsfilename', type=str, default='/nfs/projects/funcom/data/javastmt/output/coms.test')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=50000)
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)

    args = parser.parse_args()

    predfileA = args.inputA
    predfileB = args.inputB
    comsfile = args.comsfilename

    batchsize = args.batchsize
    gpu = args.gpu
    vmemlimit = args.vmemlimit

    outdir = args.outdir
    dataprep = args.dataprep
    sbt = args.sbt
    diffonly = args.diffonly
    R = args.shuffles

    comsfile = dataprep + '/' + comsfile

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    score = prep_dataset(comsfile, predfileA, predfileB, args.delim, args.batchsize, args.diffonly)
    print("\nFinal Scores Ttest:", score)