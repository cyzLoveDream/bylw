# -*- coding:utf-8 -*-

import os
import pickle
import numpy as np
import time
import re
from string import punctuation
# from nltk.stem import SnowballStemmer
def load_word_vec(path,word2ix=None):
	"""
	载入预训练的词向量
	:param path: pre-train的路径
	:param word2index:
	:return: {word:vec}
	"""
	readlines = open(path,'r',encoding="utf-8",newline="\n",errors="ignore")
	word_vec = {}
	for line in readlines:
		tokens = line.rstrip().split()
		if word2ix is None or tokens[0] in word2ix.keys():
			word_vec[tokens[0]] = np.asarray(tokens[1:],dtype="float32")
	return word_vec


def build_embedding_matrix(word2ix,embed_dim,type):
	"""
	构建预训练的词向量
	:param word2ix:
	:param embed_dim:预训练词向量的维度,不在预训练词向量中随机初始化成[-0.01,0.01)之间
	:param type: 数据集的类型
	:return:
	"""
	embedding_matrix_file_name = '../extra_data/{0}_{1}_embedding_matrix.dat'.format(str(embed_dim),type)
	if os.path.exists(embedding_matrix_file_name):
		# print("loading embedding matrix {0}:{1} ".format(embedding_matrix_file_name,
		#                                                 time.strftime("%H:%M:%S",time.localtime(time.time()))))
		embedding_matrix = pickle.load(open(embedding_matrix_file_name,'rb'))
	else:
		# print("loading word vectors...")
		embedding_matrix = 0.2 * np.random.random((len(word2ix) + 2,embed_dim)) - 0.1
		fname = '../extra_data/glove.twitter.27B.' + \
		        str(embed_dim) + "d.txt" if embed_dim != 300 else "../extra_data/glove.42B.300d.txt"
		word_vec = load_word_vec(fname, word2ix)
		# print("building embedding matrix {0}:{1}".format(embedding_matrix_file_name,
		#                                                 time.strftime("%H:%M:%S",time.localtime(time.time()))))
		for word, i in word2ix.items():
			vec = word_vec.get(word)
			if vec is not None :
				embedding_matrix[i] = vec
		pickle.dump(embedding_matrix, open(embedding_matrix_file_name,"wb"))
	return embedding_matrix


def text_to_wordlist(text,remove_stop_words=True,stem_words=False):
	# Clean the text, with the option to remove stop_words and to stem words.
	
	# Clean the text
	text = re.sub(r"[^A-Za-z0-9]"," ",text)
	text = re.sub(r"-LRB-","LRB",text)
	text = re.sub(r"-RRB-","RRB",text)
	text = re.sub(r"what's","what is ",text)
	text = re.sub(r"What's","what is ",text)
	text = re.sub(r"\'s","is",text)
	text = re.sub(r"\'ve"," have ",text)
	text = re.sub(r"can't","can not ",text)
	text = re.sub(r"ca n't","can not ",text)
	text = re.sub(r"n't"," not ",text)
	text = re.sub(r"I'm","I am",text)
	text = re.sub(r" m "," am ",text)
	text = re.sub(r"\'re"," are ",text)
	text = re.sub(r"\'d"," would ",text)
	text = re.sub(r"\'ll"," will ",text)
	text = re.sub(r"60k"," 60000 ",text)
	text = re.sub(r" e g "," eg ",text)
	text = re.sub(r" b g "," bg ",text)
	text = re.sub(r"\0s","0",text)
	text = re.sub(r" 9 11 ","911",text)
	text = re.sub(r"e-mail","email",text)
	text = re.sub(r"\s{2,}"," ",text)
	text = re.sub(r"quikly","quickly",text)
	text = re.sub(r" usa "," America ",text)
	text = re.sub(r" USA "," America ",text)
	text = re.sub(r" u s "," America ",text)
	text = re.sub(r" uk "," England ",text)
	text = re.sub(r" UK "," England ",text)
	text = re.sub(r"india","India",text)
	text = re.sub(r"switzerland","Switzerland",text)
	text = re.sub(r"china","China",text)
	text = re.sub(r"chinese","Chinese",text)
	text = re.sub(r"imrovement","improvement",text)
	text = re.sub(r"intially","initially",text)
	text = re.sub(r"quora","Quora",text)
	text = re.sub(r" dms ","direct messages ",text)
	text = re.sub(r"demonitization","demonetization",text)
	text = re.sub(r"actived","active",text)
	text = re.sub(r"kms"," kilometers ",text)
	text = re.sub(r"KMs"," kilometers ",text)
	text = re.sub(r" cs "," computer science ",text)
	text = re.sub(r" upvotes "," up votes ",text)
	text = re.sub(r" iPhone "," phone ",text)
	text = re.sub(r"\0rs "," rs ",text)
	text = re.sub(r"calender","calendar",text)
	text = re.sub(r"ios","operating system",text)
	text = re.sub(r"gps","GPS",text)
	text = re.sub(r"gst","GST",text)
	text = re.sub(r"AND","and",text)
	text = re.sub(r"programing","programming",text)
	text = re.sub(r"bestfriend","best friend",text)
	text = re.sub(r"dna","DNA",text)
	text = re.sub(r"III","3",text)
	text = re.sub(r"the US","America",text)
	text = re.sub(r"NYC","New York City",text)
	# text = re.sub(r"1-to-1","one to one",text)
	# text = re.sub(r"i5"," ",text)
	# text = re.sub(r"i7"," ",text)
	# text = re.sub(r"AGAIN","again",text)
	# text = re.sub(r"do n't","do not ",text)
	# text = re.sub(r"HP","high power ",text)
	# text = re.sub(r"pc","computer ",text)
	# text = re.sub(r"its","it is ",text)
	# text = re.sub(r"CD","Compact Disc ",text)
	# text = re.sub(r"NOT","not ",text)
	
	text = re.sub(r"Astrology","astrology",text)
	text = re.sub(r"Method","method",text)
	text = re.sub(r"Find","find",text)
	text = re.sub(r"banglore","Banglore",text)
	text = re.sub(r" J K "," JK ",text)
	
	# Remove punctuation from text
	text = ''.join([c for c in text if c not in punctuation])
	
	# Optionally, remove stop words
	if remove_stop_words:
		text = text.split()
		text = [w for w in text if not w in stop_words]
		text = " ".join(text)
	
	# Optionally, shorten words to their stems
	if stem_words:
		text = text.split()
		stemmer = SnowballStemmer('english')
		stemmed_words = [stemmer.stem(word) for word in text]
		text = " ".join(stemmed_words)
	
	# Return a list of words
	return (text)
