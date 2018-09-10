# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class DynamicLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layer=1, bias = True, batch_first = True, dropout=0,
	             bidirectional=False, only_use_last_state=False, rnn_type="LSTM", return_use_tuple = True):
		"""
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
		"""
		super(DynamicLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layer = num_layer
		self.bias = bias
		self.batch_first = batch_first
		self.bidirectional = bidirectional
		self.only_use_last_state = only_use_last_state
		self.rnn_type = rnn_type
		self.return_use_tuple = return_use_tuple
		if self.rnn_type == "LSTM":
			self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer,
			                   bias=bias, batch_first=batch_first, dropout=dropout,bidirectional=bidirectional)
		elif self.rnn_type == "GRU":
			self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer,
			                  bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
		elif self.rnn_type == "RNN":
			self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer,
			                  bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
	def forward(self, x, x_len):
		"""
		sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort
		:param x: sequence embedding vectors
		:param x_len: numpy/tensor list, 表示句子的序列长度
		:return:
		"""
		# sorted
		# x_sort_ix = np.argsort(-x_len)
		# x_unsort_ix = torch.LongTensor(np.argsort(x_sort_ix))
		# x_len = x_len(x_sort_ix)
		# x = x[torch.LongTensor(x_sort_ix)]
		
		x_sort_ix = np.argsort(-x_len)
		x_unsort_ix = torch.LongTensor(np.argsort(x_sort_ix))
		x_len = x_len[x_sort_ix]
		x = x[torch.LongTensor(x_sort_ix)]
		
		# pack
		x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
		
		# process using the selected RNN
		if self.rnn_type == "LSTM":
			out_pack, (ht, ct) = self.RNN(x_emb_p, None)
		else:
			out_pack, ht = self.RNN(x_emb_p,None)
			ct = None
		# unsort: h
		ht = torch.transpose(ht, 0, 1)[x_unsort_ix]
		ht = torch.transpose(ht, 0, 1)
		if self.only_use_last_state:
			return ht
		else:
			# unpack: out
			out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
			out = out[0]
			out = out[x_unsort_ix]
			# unsort: out
			if self.rnn_type == "LSTM":
				ct = torch.transpose(ct, 0, 1)[x_unsort_ix]
				ct = torch.transpose(ct, 0, 1)
			if self.return_use_tuple:
				return out, (ht, ct)
			else:
				return out, ht, ct
				