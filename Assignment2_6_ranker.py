import sys
import numpy as np
import csv
import heapq
import pickle
from recordclass import recordclass
import glob
import math
from numpy.linalg import norm

if len(sys.argv) != 3:
    print("Usage: python Assignment2_6_ranker.py /Data/en_BDNews24 model_queries_6.pth")
    exit(1)

data_path = sys.argv[1]
model_pth = sys.argv[2]
queries_pth = 'queries_6.txt'

NumD = 0

posting = recordclass('posting', 'term_freq doc_id')
posting_list = recordclass('posting_list', 'doc_freq postings')


def get_tf_idf(tfs, iindex, schemes, rindex):
    global NumD

    maxm = max(tfs.values()) * 1.0
    avg = float(sum(tfs.values())) / float(len(tfs))

    qvec = [0.0] * len(iindex)
    qvecs = [qvec] * len(schemes)
    sqs = [0.0] * len(schemes)

    for (term, tf2) in tfs.items():
        if term not in iindex:
            continue
        pl = iindex[term]

        for i, scheme in enumerate(schemes):
            tf = tf2
            if scheme[0] == 'l':
                tf = 1.0 + math.log(tf, 10)
            elif scheme[0] == 'L':
                tf = (1.0 + math.log(tf, 10))/(1.0 + math.log(avg, 10))
            elif scheme[0] == 'a':
                tf = 0.5 + ((0.5 * tf) / maxm)
            else:
                raise NotImplementedError

            df = pl.doc_freq
            if scheme[1] == 't':
                df = math.log(NumD/df, 10)
            elif scheme[1] == 'n':
                df = 1.0
            elif scheme[1] == 'p':
                df = max(0.0, math.log((NumD - df) / df), 10)
            else:
                raise NotImplementedError

            temp = tf * df
            qvecs[i][rindex[term]] = temp
            sqs[i] += temp * temp


    for i in range(len(schemes)):
        sqss = math.sqrt(sqs[i])
        if sqss > 0:
            qvecs[i] = np.array(qvecs[i])/sqss

    return qvecs


def get_tfs(txt):
    tfs = {}
    for term in txt:
        if term not in tfs:
            tfs[term] = 0.0
        tfs[term] += 1.0

    return tfs

import time
if __name__ == "__main__":
    iindex = {}

    with open(model_pth, 'rb') as handle:
        iindex = pickle.load(handle)

    rindex = {k: i for i, k in enumerate(iindex.keys())}

    qschemes = ["ltc", "Lpc", "apc"]
    dschemes = ["lnc", "Lnc", "anc"]

    for filename in glob.glob(data_path + "**/*/*", recursive=True):
        NumD += 1
    qvecs = {}

    with open(queries_pth, 'r') as fp:
        for line in fp.read().splitlines():
            qid, qtext = line.split(",")
            qtext = qtext.split(" ")

            qvecs[qid] = {}
            tfs = get_tfs(qtext)
            qvecs[qid] = get_tf_idf(tfs, iindex, qschemes, rindex)

    dtfs = {}

    for (term, pl) in iindex.items():
        for posting in pl.postings:
            if posting.doc_id not in dtfs:
                dtfs[posting.doc_id] = {}
            dtfs[posting.doc_id][term] = posting.term_freq

    def cos_sim(a, b): return (np.dot(a, b) / (norm(a) * norm(b)))
    cnt = 0

    tic = time.perf_counter()
    scores = [{}, {}, {}]

    for (doc_id, v) in dtfs.items():
        cnt += 1
        dvecs = get_tf_idf(v, iindex, dschemes, rindex)

        for i in range(3):
            for (qid, qvec) in qvecs.items():
                if qid not in scores[i]:
                    scores[i][qid] = {}
                scores[i][qid][doc_id] = cos_sim(
                    qvec[i], dvecs[i])

        if cnt % 5000 == 0:
            tic2 = time.perf_counter()
            print("Completed "+str(cnt)+" docs ")
            print(f"total Took  {tic2 - tic:0.4f} seconds")

    outfiles = ["Assignment2_6_ranked_list_A.csv",
                "Assignment2_6_ranked_list_B.csv", "Assignment2_6_ranked_list_C.csv"]

    cnt = -1
    for outf in outfiles:
        cnt += 1
        with open(outf, 'w') as fp:

            writer = csv.DictWriter(fp, fieldnames=["Query_ID", "Document_ID"])
            writer.writeheader()

            writer = csv.writer(fp)
            for qid in qvecs.keys():
                most_relv = heapq.nlargest(50, scores[cnt][qid], scores[cnt][qid].get)
                for did in most_relv:
                    writer.writerows([[qid, did]])

