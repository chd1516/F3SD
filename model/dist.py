import torch
import torch.nn as nn
import torch.nn.functional as F
class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, temp=1.0, rank=64):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.rank = rank
        self.temp =temp
    def cosine_similarity(self, a, b, eps=1e-8):
        return (a*b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

    def correlation(self, s, t, eps=1e-8):
        return self.cosine_similarity(s-s.mean(1).unsqueeze(1), t-t.mean(1).unsqueeze(1), eps)
    def inter_class(self, s, t):
        return 1 - self.correlation(s, t).mean()
    def intra_class(self, s ,t):
        return self.inter_class(s.transpose(0,1), t.transpose(0,1))
    def forward(self, s, t, **kwargs):
        s = (s / self.temp).softmax(dim=1)
        t = (t / self.temp).softmax(dim=1)
        inter_loss = self.temp ** 2 * self.inter_class(s, t)
        intra_loss = self.temp ** 2 * self.intra_class(s, t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss
