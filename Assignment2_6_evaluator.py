import numpy as np
import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python Assignment2_6_evaluator.py ./Data/rankedRelevantDocList.csv Assignment2_11_ranked_list_K.csv")
    exit(1)

gold_pth = sys.argv[1]
ranked_pth = sys.argv[2]
typ = ranked_pth[-5]

# For a single query P@k
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

# For a single query AP @n
def average_precision(r, n):
    r = r[:n]
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

# For a single query DCG@k
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

# For a single query NDCG@k
def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def df_to_dict(df, isgold):
    df2 = df.groupby("Query_ID")
    ret = {}
    for x, y in df2:
        if isgold:
            ret[x] = dict(zip(y["Document_ID"].to_list(),
                              y["Relevance_Score"].to_list()))
        else:
            ret[x] = y["Document_ID"].to_list()

    return ret


def relevance(rank, gold, binary=False):
    myl = []
    for doc in rank:
        if doc not in gold:
            myl.append(0)
        else:
            if binary:
                myl.append(1)
            else:
                myl.append(gold[doc])

    return myl


if __name__ == '__main__':
    gold = pd.read_csv(open(gold_pth, 'r'))
    ranked = pd.read_csv(open(ranked_pth, 'r'))
    gold_d = df_to_dict(gold, True)
    ranked_d = df_to_dict(ranked, False)

    results = []
    for (qid, v) in gold_d.items():
        myl = relevance(ranked_d[qid], gold_d[qid], binary=True)
        ap10 = average_precision(myl, 10)
        ap20 = average_precision(myl, 20)

        myl = relevance(ranked_d[qid], gold_d[qid])
        ndcg10 = ndcg_at_k(myl, 10)
        ndcg20 = ndcg_at_k(myl, 20)

        results.append((qid, ap10, ap20, ndcg10, ndcg20))

    out = "Assignment2_6_metrics_" + str(typ) + ".csv"
    df = pd.DataFrame(results, columns=[
                      "Query_ID", "AP@10", "AP@20", "NDCG@10", "NDCG@20"])
    df.to_csv(out,float_format='%.3f', index = False)

    means = df.mean(axis=0)

    with open(out, 'a') as fp:
        fp.write("\n\nOverall metrics\n")
        fp.write("\nmAP@10 = "+str(means["AP@10"]))
        fp.write("\nmAP@20 = "+str(means["AP@20"]))
        fp.write("\naverNDCG@10 = "+str(means["NDCG@10"]))
        fp.write("\naverNDCG@20 = "+str(means["NDCG@20"]))

