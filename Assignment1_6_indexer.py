import nltk
import pickle
import glob
import os
import re
import sys
import lxml.etree as ET
# import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from recordclass import recordclass

if len(sys.argv) != 2:
	print("Usage: python Assignment1_6_indexer.py /Data/en_BDNews24")
	exit(1)

root_dir = sys.argv[1]

def get_text(filename):
	with open(filename, 'r') as fp:
		try:
			return re.search('<TEXT>(.*)</TEXT>', ''.join(fp.read().splitlines())).group(1)
		except Exception as e:
			print(e)
			print("Error parsing " + filename)
			exit(1)
			return ""

def preprocess(txt):
	# tokenize
	tokens = word_tokenize(txt)

	# remove punctuation from each word
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]

	# convert to lower case
	stripped = [w.lower() for w in stripped]

	# remove stop words
	stop_words = set(stopwords.words('english'))
	stop_words.add("")
	words = [w for w in stripped if not w in stop_words]

	# lemmatize
	lemma_function = WordNetLemmatizer()
	mod_tokens = [lemma_function.lemmatize(tok) for tok in words]

	return mod_tokens

posting = recordclass('posting', 'term_freq doc_id')
posting_list = recordclass('posting_list', 'doc_freq postings')

def add_tokens(iindex, tokens, doc_id):
	tf = {}
	for tok in tokens:
		if tok not in tf:
			tf[tok] = 0
		tf[tok] += 1

	for (k, term_freq) in tf.items():
		if k not in iindex:
			iindex[k] = posting_list(0, [])
		iindex[k].postings.append(posting(term_freq, doc_id))
		iindex[k].doc_freq += 1

if __name__ == "__main__":	
	iindex = {}

	for filename in glob.glob(root_dir + "**/*/*", recursive = True):
		fname = os.path.basename(filename)
		txt = get_text(filename)
		tokens = preprocess(txt)
		add_tokens(iindex, tokens, fname)

	for (k, v) in iindex.items():
		iindex[k].postings = sorted(v.postings, key = lambda x: x.doc_id)

	print(len(iindex))

	with open('model_queries_6.pth', 'wb') as handle:
	    pickle.dump(iindex, handle, protocol=pickle.HIGHEST_PROTOCOL)

