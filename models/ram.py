# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from models.BasicModule import BasicModule
from layers.DynamicLSTM import DynamicLSTM
from layers.attention import Attention
class RAM(BasicModule):
	def __init__(self, embedding_matrix, opt):
		super(RAM, self).__init__()
		self.model_name = "ram"
		self.opt = opt
		self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
		self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim,num_layer=1,
		                                   batch_first=True,bidirectional=True)
		self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layer=1,
		                                  batch_first=True, bidirectional=True)
		self.attention = Attention(opt.hidden_dim * 2, score_function="mlp")
		self.gru_cell = nn.GRUCell(opt.hidden_dim * 2, opt.hidden_dim * 2)
		self.dense = nn.Linear(opt.hidden_dim *2, opt.polarity_dim)
		
	def locationed_memory(self, memory, memory_len):
		# here we just simply calculate the location vector in Model2's manner
		for i in range(memory.size(0)):
			for ix in range(memory_len[i]):
				memory[i][ix] *= (1-float(ix) / int(memory_len[i]))
		return memory
	
	def forward(self, inputs):
		text_raw_indices, aspect_indices = inputs[0], inputs[1]
		memory_len = torch.sum(text_raw_indices != 0, dim=-1)
		aspect_len = torch.sum(aspect_indices != 0, dim=-1)
		nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
		
		memory = self.embed(text_raw_indices)
		memory, (_,_) = self.bi_lstm_context(memory, memory_len)
		# memory = self.locationed_memory(memory, memory_len)
		aspect =  self.embed(aspect_indices)
		aspect, (_,_) = self.bi_lstm_aspect(aspect, aspect_len)
		aspect = torch.sum(aspect, dim=1)
		aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0),1))
		et = aspect
		for _ in range(self.opt.hops):
			it_al = self.attention(memory, et).squeeze(dim=1)
			et = self.gru_cell(it_al, et)
			del it_al
		del memory
		out = self.dense(et)
		return out