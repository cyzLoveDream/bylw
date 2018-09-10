# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.BasicModule import BasicModule


class MYNETWork(BasicModule):
    def __init__(self, embedding_matrix, opt):
        super(MYNETWork, self).__init__()
        self.model_name = "mynetwork"
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
        self.convs_left = nn.ModuleList([nn.Conv1d(embedding_matrix.shape[1],opt.fillter,K) for K in [int(x) for x in opt.kernel_sizes]])
        self.convs_right = nn.ModuleList([nn.Conv1d(embedding_matrix.shape[1], opt.fillter, k) for k in [int(x) for x in opt.kernel_sizes]])
        self.btNorm = nn.BatchNorm1d(opt.fillter)
        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(len(opt.kernel_sizes) * opt.fillter*2, opt.polarity_dim)
        self.fc_aspect = nn.Linear(embedding_matrix.shape[1], opt.fillter)
        self.opt = opt

    def forward(self, inputs):
        left_indices_with_aspect, right_indices_with_aspect, aspect_indices = inputs[0], inputs[1], inputs[2]

        left_feature =self.drop(self.embed(left_indices_with_aspect))
        right_feature = self.drop(self.embed(right_indices_with_aspect))

        # 对多个aspect的词做一个平均
        aspect_v = self.embed(aspect_indices)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x_left = [F.tanh(conv(left_feature.transpose(1,2))) for conv in self.convs_left]
        y_left = [F.relu(conv(left_feature.transpose(1,2))) + self.fc_aspect(aspect_v).unsqueeze(2) for conv in self.convs_left]
        # y_left = [self.fc_aspect(aspect_v).unsqueeze(2)] * len(self.opt.kernel_sizes)
        x_right = [F.tanh(conv(right_feature.transpose(1,2))) for conv in self.convs_right]
        y_right = [F.relu(conv(right_feature.transpose(1,2))) + self.fc_aspect(aspect_v).unsqueeze(2) for conv in self.convs_right]
        # y_right = [self.fc_aspect(aspect_v).unsqueeze(2)] * len(self.opt.kernel_sizes)
        x_l = [i * j for i,j in zip(x_left,y_left)]
        x_r = [i * j for i, j in zip(x_right,y_right)]

        # pooling method
        xl = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x_l]  # [(N,Co), ...]*len(Ks)
        xl = [i.view(i.size(0),-1) for i in xl]
        xl = torch.cat(xl,1)

        xr = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x_r]  # [(N,Co), ...]*len(Ks)
        xr = [i.view(i.size(0),-1) for i in xr]
        xr = torch.cat(xr,1)

        # x = torch.mul(xr, xl)
        x = torch.cat((xl,xr),1)
        x = F.dropout(x, p=0.2)
        logit = self.fc1(x)  # (N,C)
        return logit
