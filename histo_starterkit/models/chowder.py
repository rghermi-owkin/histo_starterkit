import torch
from torch.nn import Module
import torch.nn as nn

class Chowder(Module):

    def __init__(self, feature_dim=2048, n_top=10, n_bottom=10):
        super(Chowder, self).__init__()
        self.feature_dim = feature_dim
        self.n_top = n_top
        self.n_bottom = n_bottom
        
        self.attention_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, 
                      out_channels=1,
                      kernel_size=1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(self.n_top+self.n_bottom, 1),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        
        scores = self.attention_layer(x)
        scores = scores.squeeze(1)
        
        top, top_idx = scores.topk(k=self.n_top, dim=1)
        bottom, bottom_idx = scores.topk(
            k=self.n_bottom, largest=False, dim=1)
        extreme_scores = torch.cat([top, bottom], dim=1)
        #indices = torch.cat([top_idx, bottom_idx], dim=1)
        
        prediction = self.classification_head(extreme_scores)
        
        return prediction, scores