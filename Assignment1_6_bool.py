import nltk
import pickle
import glob
import os
import re
import sys

from queue import PriorityQueue
from recordclass import recordclass

if len(sys.argv) != 3:
	print("Usage: python Assignment1_6_indexer.py ./model_queries_6.pth ./queries_6.txt")
	exit(1)

model_pth = sys.argv[1]
queries_pth = sys.argv[2]

posting = recordclass('posting', 'term_freq doc_id')
posting_list = recordclass('posting_list', 'doc_freq postings')

def merge_postings(l1, l2):
	m = len(l1)
	n = len(l2)

	i = 0
	j = 0
	out = []
	while i < m and j < n:
		if l1[i] == l2[j]:
			out.append(l1[i])
			i += 1
			j += 1
		elif l1[i] < l2[j]:
			i += 1
		else:
			j += 1

	return out

def merge_all2(iindex, qtext):
	res = None
	for qt in qtext:
		if qt not in iindex:
			return []

		x = set([x.doc_id for x in iindex[qt].postings])
		if res is None:
			res = x
		else:
			res = res & x

	return list(res)

def merge_all(iindex, qtext):
	pq = PriorityQueue()

	for qt in qtext:
		if qt not in iindex:
			return []

		pq.put((iindex[qt].doc_freq, [x.doc_id for x in iindex[qt].postings]))

	while pq.qsize() != 1:
		item1 = pq.get()
		item2 = pq.get()
		merged = merge_postings(item1[1], item2[1])
		
		pq.put((len(merged), merged))

	out = pq.get()
	return out[1]

if __name__ == "__main__":

	iindex = {}

	with open(model_pth, 'rb') as handle:
	    iindex = pickle.load(handle)

	with open('Assignment1_6_results.txt', 'w') as fpr:
		with open(queries_pth, 'r') as fp:
			for line in fp.read().splitlines():
				qid, qtext = line.split(",")
				qtext = qtext.split(" ")
				qtext = list(set(qtext)) # distinct query terms

				out = merge_all(iindex, qtext)
				fpr.write(str(qid) + ":" + " ".join(out) + "\n")
