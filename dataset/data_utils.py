# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
from torch.utils.data import Dataset
from utils.utils import build_embedding_matrix
from utils.utils import text_to_wordlist
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

class Tokenizer(object):
	"""
	将语料序列化
	"""
	def __init__(self,lower=False,max_seq_len=None,max_aspect_len=None,char_level=None):
		self.lower = lower
		self.max_seq_len = max_seq_len
		self.max_aspect_len = max_aspect_len
		self.char_level = char_level
		self.word2ix = {}
		self.ix2word = {}
		self.ix = 1
	
	def fit_on_text(self,text):
		"""
		所有文本形成的一个超长字符串
		:param text:
		:return:
		"""
		if self.lower:
			text = text.lower()
		words = text.split()
		for w in words:
			if w not in self.word2ix:
				self.word2ix[w] = self.ix
				self.ix2word[self.ix] = w
				self.ix += 1
	
	@staticmethod
	def pad_sequence(sequence,maxlen,dtype="int64",padding="pre",truncating="pre",value=0.):
		x = (np.ones(maxlen) * value).astype(dtype)
		if truncating == "pre":
			# 前向截断
			trunc = sequence[-maxlen:]
		else:
			# 后向截断
			trunc = sequence[:maxlen]
		trunc = np.asarray(trunc,dtype=dtype)
		if padding == "post":
			# 后向填充
			x[:len(trunc)] = trunc
		else:
			# 前向填充
			x[-len(trunc):] = trunc
		return x
	
	def text_to_sequence(self,text,reverse=False):
		if self.lower:
			text = text.lower()
		words = text.split()
		unknowix = len(self.word2ix) + 1
		sequence = [self.word2ix[w] if w in self.word2ix else unknowix for w in words]
		if len(sequence) == 0:
			sequence = [0]
		pad_and_trunc = "post"
		if reverse:
			sequence = sequence[::-1]
		return Tokenizer.pad_sequence(sequence,self.max_seq_len,dtype="int64",padding=pad_and_trunc,
		                              truncating=pad_and_trunc)
	
class ABSADataset(Dataset):
	def __init__(self, data):
		self.data = data
		
	def __getitem__(self, index):
		return self.data[index]
	
	def __len__(self):
		return len(self.data)
	
class ABSADatasetReader:
	@staticmethod
	def __read_text__(fname):
		fin = open(fname, 'r', encoding="utf-8", newline="\n", errors="ignore")
		lines = fin.readlines()
		fin.close()
		text = ''
		for i in range(0, len(lines), 3):
			text_left, _, text_right = [s.encode("utf-8").decode("utf-8-sig").strip() for s in lines[i].partition("$T$")]
			aspect = lines[i+1].strip()
			text_raw = text_left + " " + aspect + " " + text_right
			text += text_raw
		return text
	
	@staticmethod
	def __read_data__(fname, tokenizer):
		fin = open(fname, 'r', encoding="utf-8", newline="\n", errors="ignore")
		lines = fin.readlines()
		all_data = []
		for i in range(0, len(lines), 3):
			# lines[i] = text_to_wordlist(lines[i],remove_stop_words=False,stem_words=False)
			text_left,_, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
			aspect = lines[i+1].lower().strip()
			polarity = lines[i+2].strip()
			text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
			text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
			text_left_indices = tokenizer.text_to_sequence(text_left)
			text_left_with_aspect = tokenizer.text_to_sequence(text_left + " " + aspect)
			text_right_indices = tokenizer.text_to_sequence(text_right)
			text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right)
			aspect_indices = tokenizer.text_to_sequence(aspect)
			polarity = int(polarity) + 1
			data = {
				"text_raw_indices":text_raw_indices,
				"text_raw_without_aspect_indices":text_raw_without_aspect_indices,
				"text_left_indices": text_left_indices,
				"text_left_with_aspect_indices": text_left_with_aspect,
				"text_right_indices": text_right_indices,
				"text_right_with_aspect_indices":text_right_with_aspect_indices,
				"aspect_indices":aspect_indices,
				"polarity":polarity
				}
			all_data.append(data)
		pls = []
		for i in all_data:
			pls.append(i.get("polarity"))
		# print("the all data is {0}, the POS is {1}, the NEU is {2}, the NEG is {3}".format(len(all_data), pls.count(2),pls.count(1),pls.count(0)))
		return all_data
	
	def __init__(self, dataset="bylw",embed_dim = 100, max_seq_len = 80):
		# print("preparing {0} dataset...".format(dataset))
		fname = {
			"bylw":{
				"train": "../data_set/train.csv",
				"valid":"../data_set/valid.csv",
				"test": "../data_set/test.csv"
				}
			}
		text = ABSADatasetReader.__read_text__(fname[dataset]["train"])
		tokenizer = Tokenizer(max_seq_len = max_seq_len)
		tokenizer.fit_on_text(text.lower())
		self.embedding_matrix = build_embedding_matrix(tokenizer.word2ix, embed_dim,dataset)
		if dataset in ["my-laptop","my-restaurants"]:
			self.train_data = ABSADataset(ABSADatasetReader.__read_data__(fname[dataset]["train"],tokenizer))
			self.test_data = ABSADataset(ABSADatasetReader.__read_data__(fname[dataset]["test"], tokenizer))
		else:
			self.train_data = ABSADataset(ABSADatasetReader.__read_data__(fname[dataset]["train"],tokenizer))
			self.valid_data = ABSADataset(ABSADatasetReader.__read_data__(fname[dataset]["valid"], tokenizer))
			self.test_data = ABSADataset(ABSADatasetReader.__read_data__(fname[dataset]["test"], tokenizer))
