# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import warnings
from layers.attention import Attention
from layers.DynamicLSTM import DynamicLSTM
from models.BasicModule import BasicModule

warnings.filterwarnings("ignore")

class InterAttNet(BasicModule):
	def __init__(self, embedding_matrix, opt):
		super(InterAttNet, self).__init__()
		self.model_name = "IAN"
		self.opt = opt
		self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
		self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layer=1, batch_first=True,return_use_tuple=True)
		self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layer=1, batch_first=True, return_use_tuple=True)
		self.attention_aspect = Attention(opt.hidden_dim, score_function="bi_linear", dropout=0.5)
		self.attention_context = Attention(opt.hidden_dim, score_function="bi_linear", dropout=0.5)
		self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarity_dim)

		
	def forward(self, inputs):
		text_raw_indices, aspect_indices = inputs[0], inputs[1]
		text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
		aspect_len = torch.sum(aspect_indices != 0, dim=-1)
		
		context = self.embed(text_raw_indices)
		aspect = self.embed(aspect_indices)
		context, (_,_) = self.lstm_context(context, text_raw_len)
		aspect, (_,_) = self.lstm_context(aspect, aspect_len)
		
		aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
		aspect = torch.sum(aspect, dim = 1)
		aspect = torch.div(aspect, aspect_len.view(aspect_len.size(0), 1))
		
		text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
		context = torch.sum(context, dim=1)
		context = torch.div(context, text_raw_len.view(text_raw_len.size(0),1))
		
		
		
		aspect_final = self.attention_aspect(aspect, context).squeeze(dim=1)
		context_final = self.attention_context(context, aspect).squeeze(dim=1)
		
		x = torch.cat((aspect_final, context_final), dim=-1)
		x = nn.Dropout(0.5)(x)
		out = self.dense(x)
		
		return out