import torch
from torch.nn import Module
import torch.nn as nn

class DeepMIL(Module):

    def __init__(self, feature_dim=2048):
        super(DeepMIL, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, 
                      out_channels=1,
                      kernel_size=1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        
        attention_weights = self.attention_layer(x)
        attention_weights = attention_weights.squeeze(1)
        norm = attention_weights.norm(p=2, dim=1, keepdim=True)
        attention_weights = attention_weights.div(norm)
        attention_weights = attention_weights.unsqueeze(2)
        
        aggregated_feature = x @ attention_weights
        aggregated_feature = aggregated_feature.squeeze(2)
        
        prediction = self.classification_head(aggregated_feature)
        
        return prediction, aggregated_feature