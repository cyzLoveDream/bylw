# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from layers.DynamicLSTM import DynamicLSTM
from models.BasicModule import BasicModule

class LSTM(BasicModule):
	def __init__(self, embedding_matrix, opt):
		super(LSTM, self).__init__()
		self.model_name = "lstm"
		self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
		self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim,num_layer=1, batch_first=True)
		self.dense = nn.Linear(opt.hidden_dim, opt.polarity_dim)
		
	def forward(self, inputs):
		text_raw_indices = inputs[0]
		x = self.embed(text_raw_indices)
		x_len = torch.sum(text_raw_indices !=0, dim=-1)
		_, (h_n,_) = self.lstm(x, x_len)
		out = self.dense(h_n[0])
		return out