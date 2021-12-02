import torch
from torch.nn import Module
import torch.nn as nn

class MeanPool(Module):

    def __init__(self, feature_dim=2048):
        super(MeanPool, self).__init__()
        self.feature_dim = feature_dim
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
        )

    def forward(self, x):
        
        # Feature aggregation
        aggregated_feature = torch.mean(x, dim=1) # need to take care of zero-padding
        
        # Classification head
        prediction = self.classification_head(aggregated_feature)
        
        # Tests
        #assert x.shape == (batch_size, max_tiles, feature_dim)
        #assert aggregated_feature.shape == (batch_size, feature_dim)
        #assert prediction.shape == (batch_size, 1)
        #assert np.linalg.norm(np.mean(np.array(x), axis=1) - np.array(aggregated_feature)) < 1e-3

        return prediction, aggregated_feature
    
class MaxPool(Module):

    def __init__(self, feature_dim=2048):
        super(MaxPool, self).__init__()
        self.feature_dim = feature_dim
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
        )

    def forward(self, x):
        
        # Feature aggregation
        aggregated_feature, _ = torch.max(x, dim=1)
        
        # Classification head
        prediction = self.classification_head(aggregated_feature)

        #assert x.shape == (4, 1000, 2048)
        #assert aggregated_feature.shape == (4, 2048)
        #assert prediction.shape == (4, 1)
        #assert np.linalg.norm(np.max(np.array(x), axis=1) - np.array(aggregated_feature)) < 1e-3

        return prediction, aggregated_feature
    
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
        bottom, bottom_idx = scores.topk(k=self.n_bottom, largest=False, dim=1)
        extreme_scores = torch.cat([top, bottom], dim=1)
        #indices = torch.cat([top_idx, bottom_idx], dim=1)
        
        prediction = self.classification_head(extreme_scores)
        
        return prediction, scores