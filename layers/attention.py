# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
import numpy as np
np.random.seed(1024)
class Attention(nn.Module):
	def __init__(self,embed_dim,hidden_dim=None,head=1,score_function="scaled_dot_product",dropout=0.1):
		"""
		注意力机制
		:param embed_dim:
		:param hidden_dim:
		:param head: 多头注意力的头数
		:param score_function: scaled_dot_product / mlp(concat) / bi-linear(general dot),计算得分的方式
		:param dropout:
		"""
		super(Attention,self).__init__()
		if hidden_dim is None:
			hidden_dim = embed_dim // head
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.head = head
		self.score_function = score_function
		self.w_kx = nn.Parameter(torch.FloatTensor(head,embed_dim,hidden_dim))
		self.w_qx = nn.Parameter(torch.FloatTensor(head,embed_dim,hidden_dim))
		self.proj = nn.Linear(head * hidden_dim, embed_dim)
		self.dropout = nn.Dropout(dropout)
		if score_function == "mlp":
			self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2,1))
		elif score_function == "bi_linear":
			self.weight = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim))
		else:
			self.register_parameter("weight",None)
		self.reset_parameters()
	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(0))
		self.weight.data.uniform_(-stdv, stdv)
		qx_stdv = 1. / math.sqrt(self.w_qx.size(0))
		self.w_qx.data.uniform_(-qx_stdv, qx_stdv)
		wx_stdv = 1. / math.sqrt(self.w_kx.size(0))
		self.w_kx.data.uniform_(-wx_stdv, wx_stdv)
	
	def forward(self,k,q):
		if len(q.shape) == 2:
			q = torch.unsqueeze(q,dim=1)
		if len(k.shape) == 2:
			k = torch.unsqueeze(k,dim=1)
		mb_size = k.shape[0]
		k_len = k.shape[1]
		q_len = q.shape[1]
		
		kx = k.repeat(self.head,1,1).view(self.head,-1,self.embed_dim)
		qx = q.repeat(self.head,1,1).view(self.head,-1,self.embed_dim)
		kx = torch.bmm(kx,self.w_kx).view(-1,k_len,self.hidden_dim)
		qx = torch.bmm(qx,self.w_qx).view(-1,q_len,self.hidden_dim)
		if self.score_function == "scaled_dot_product":
			kt = kx.permute(0,2,1)
			qkt = torch.bmm(qx,kt)
			score = torch.div(qkt,math.sqrt(self.hidden_dim))
		elif self.score_function == "mlp":
			kxx = torch.unsqueeze(kx,dim=1).expand(-1,q_len,-1,-1)
			qxx = torch.unsqueeze(qx,dim=2).expand(-1,-1,k_len,-1)
			kq = torch.cat([kxx,qxx],dim=-1)
			score = F.tanh(torch.matmul(kq,self.weight).squeeze(dim=-1))
		elif self.score_function == "bi_linear":
			qw = torch.matmul(qx,self.weight)
			kt = kx.permute(0,2,1)
			score = torch.bmm(qw,kt)
		else:
			raise RuntimeError("invalid score function")
		score = F.softmax(score,dim=-1)
		output = torch.bmm(score,kx)
		output = torch.cat(torch.split(output,mb_size,dim=0),dim=-1)
		output = self.proj(output)
		output = self.dropout(output)
		return output


class SelfAttention(Attention):
	"""
	q is a parameter
	"""
	
	def __init__(self,embed_dim,hidden_dim=None,head=1,score_function="scaled_dot_product",
	             q_len=1,dropout=0.1):
		super(SelfAttention,self).__init__(embed_dim,hidden_dim,head, score_function,dropout)
		self.q_len = q_len
		self.q = nn.Parameter(torch.FloatTensor(q_len, embed_dim))
	
	def forward(self,k, **kwargs):
		mb_size = k.shape[0]
		q = self.q.expand(mb_size, -1, -1)
		return super(SelfAttention, self).forward(k, q)
