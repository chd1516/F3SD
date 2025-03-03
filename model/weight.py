import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightLearningModule(nn.Module):
    def __init__(self,input_dim):
        super(WeightLearningModule, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Sigmoid()
                )

    def forward(self, img_feats ,img_feats_n_top, img_feats_n_top_sm):
        combined_feats = torch.cat([img_feats,img_feats_n_top,img_feats_n_top_sm], dim=1)
        weights = self.fc(combined_feats)
        alpha, beta = weights[:,0].unsqueeze(1), weights[:,1].unsqueeze(1)
        return alpha, beta
