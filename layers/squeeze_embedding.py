# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class SqueezeEmbeeding(nn.Module):
	
	def __init__(self, batch_first = True):
		super(SqueezeEmbeeding, self).__init__()
		self.batch_first = batch_first
		
		
	def forward(self, x, x_len):
		"""
		在一个batch中把所有的sequence 根据batch中最长的sequence按照向量压缩
		:param x:
		:param x_len: numpy/ tensor/ list
		:return:
		"""
		# """sort"""
		x_sorted_idx = np.argsort(-x_len)
		x_unsort_idx = torch.LongTensor(np.argsort(x_sorted_idx))
		x_len = x_len[x_sorted_idx]
		x = x[torch.LongTensor(x_sorted_idx)]
		
		# pack
		x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
		# unpack
		out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
		out = out[0]
		# unsort
		out = out[x_unsort_idx]
		return out