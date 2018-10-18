# -*- coding:utf-8 -*-

import os
import pickle
import numpy as np
import time
import re
from string import punctuation
# from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
# def load_word_vec(path,word2ix=None):
# 	"""
# 	载入预训练的词向量
# 	:param path: pre-train的路径
# 	:param word2index:
# 	:return: {word:vec}
# 	"""
# 	readlines = open(path,'r',encoding="utf-8",newline="\n",errors="ignore")
# 	word_vec = {}
# 	c = 0
# 	for line in readlines:
# 		if c > 1:
# 			tokens = line.split()
# 			if word2ix is None or tokens[0] in word2ix.keys():
# 				word_vec[tokens[0]] = np.asarray(tokens[1:],dtype="float32")
# 		c += 1
# 	return word_vec


def build_embedding_matrix(word2ix,embed_dim,type,fname):
	"""
	构建预训练的词向量
	:param word2ix:
	:param embed_dim:预训练词向量的维度,不在预训练词向量中随机初始化成[-0.01,0.01)之间
	:param type: 数据集的类型
	:return:
	"""
	embedding_matrix_file_name = '../cache/{0}_{1}_embedding_matrix.dat'.format(str(embed_dim),type)
	if os.path.exists(embedding_matrix_file_name):
		# print("loading embedding matrix {0}:{1} ".format(embedding_matrix_file_name,
		#                                                 time.strftime("%H:%M:%S",time.localtime(time.time()))))
		embedding_matrix = pickle.load(open(embedding_matrix_file_name,'rb'))
	else:
		# print("loading word vectors...")
		embedding_matrix = 0.2 * np.random.random((len(word2ix) + 2,embed_dim)) - 0.1
		word_vec = Word2Vec.load(fname)
		# print("building embedding matrix {0}:{1}".format(embedding_matrix_file_name,
		#                                                 time.strftime("%H:%M:%S",time.localtime(time.time()))))
		for word, i in word2ix.items():
			if word in word_vec:
				embedding_matrix[i] = word_vec[word]
		pickle.dump(embedding_matrix, open(embedding_matrix_file_name,"wb"))
	return embedding_matrix


