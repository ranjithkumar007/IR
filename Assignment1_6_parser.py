import nltk
import pickle
import glob
import os
import re
import sys
# import lxml.etree as ET
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from recordclass import recordclass, RecordClass

if len(sys.argv) != 2:
	print("Usage: python Assignment1_6_indexer.py /Data/raw_query.txt")
	exit(1)

filename = sys.argv[1]

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

	return ' '.join(mod_tokens)

if __name__ == "__main__":
	queries = []

	root = ET.parse(filename).getroot()
	for type_tag in root.findall('top'):
		queries.append((type_tag.find('num').text, preprocess(type_tag.find('title').text)))
	
	with open('queries_6.txt', 'w') as fp:
		for query in queries:
			fp.write(query[0] + "," + query[1] + "\n")
