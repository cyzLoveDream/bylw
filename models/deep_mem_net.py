# -*- coding:utf-8 -*-

from layers.attention import Attention
import torch
import torch.nn as nn
from models.BasicModule import BasicModule
from layers.squeeze_embedding import SqueezeEmbeeding

class DeepMemNet(BasicModule):
	def locationed_memory(self, memory, memory_len, left_len, aspect_len):
		"""
		content 和aspect的位置差异
		:param memory:
		:param memory_len:
		:param left_len:
		:param aspect_len:
		:return:
		"""
		for i in range(memory.size(0)):
			for ix in range(memory_len[i]):
				aspect_start = left_len[i] - aspect_len[i]
				if ix < aspect_start:
					l = aspect_start.item() - ix
				else:
					l = ix + 1 - aspect_start.item()
				memory[i][ix] *= (1-float(l)) / int(memory_len[i])
		return memory

	def __init__(self, embedding_matrix, opt):
		super(DeepMemNet,self).__init__()
		self.model_name = "DeepMemNet"
		self.opt = opt
		self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
		self.squeeze_embedding = SqueezeEmbeeding(batch_first=True)
		self.attention = Attention(opt.embed_dim, score_function="mlp")
		self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)
		self.dense = nn.Linear(opt.embed_dim, opt.polarity_dim)

	def forward(self, inputs):
		text_raw_without_aspect_indices, aspect_indices, left_with_aspect_indices = inputs[0], inputs[1],inputs[2]

		memory_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
		aspent_len = torch.sum(aspect_indices != 0, dim=-1)
		nonzeros_aspect = torch.tensor(aspent_len, dtype=torch.float).to(self.opt.device)
		memory = self.embed(text_raw_without_aspect_indices)
		memory = self.squeeze_embedding(memory, memory_len)

		aspect = self.embed(aspect_indices)
		aspect = torch.sum(aspect, dim=1)
		aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0),1))
		x = aspect.unsqueeze(dim=1)

		for _ in range(self.opt.hops):
			x = self.x_linear(x)
			out_at = self.attention(memory, x)
			x = out_at + x
		x = x.view(x.size(0), -1)
		out = self.dense(x)
		return out
