import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule

class CNN_Gate_Aspect_Text(BasicModule):
    def __init__(self,context_embedding_matrix,opt):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.model_name = "cnn_gate"
        self.opt = opt
        D = opt.embed_dim
        Co = opt.fillter
        Ks = [int(x) for x in opt.kernel_sizes]

        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(context_embedding_matrix))
        # self.embed.weight = nn.Parameter(torch.FloatTensor(context_embedding_matrix),requires_grad=True)

        # self.aspect_embed = nn.Embedding.from_pretrained(torch.FloatTensor(context_embedding_matrix))
        # self.aspect_embed.weight = nn.Parameter(context_embedding_matrix,requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D,Co,K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D,Co,K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks) * Co, opt.polarity_dim)
        self.fc_aspect = nn.Linear(context_embedding_matrix.shape[1], Co)

    def forward(self,inputs):
        feature,aspect = inputs[0],inputs[1]
        feature = self.embed(feature)  # (N, L, D)
        aspect_v = self.embed(aspect)  # (N, L', D)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)

        x = [F.tanh(conv(feature.transpose(1,2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1,2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i,j in zip(x,y)]

        # pooling method
        x0 = [F.max_pool1d(i,i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0),-1) for i in x0]

        x0 = torch.cat(x0,1)
        logit = self.fc1(x0)  # (N,C)
        return logit