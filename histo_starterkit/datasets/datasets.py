import numpy as np

import torch
from torch.utils.data import Dataset

from histo_starterkit.utils import load_features

class SlideFeaturesDataset(Dataset):
    
    def __init__(self, df, max_tiles=1000, feature_dim=2048):
        self.df = df
        self.max_tiles = max_tiles
        self.feature_dim = feature_dim
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get label
        label = row.target
        
        # Get features
        slide_name = row.slide_name
        features = load_features(slide_name)
        
        # Sub-sample features
        ## Fixed random sub-sampling
        features = features[:self.max_tiles, 3:] # keep coordinates and discard them in the model?
        ## Variable random sub-sampling
        ## sort of data augmentation (different tiles selected for the same slide)
        ## need to handle inference time (fixed sampling? ensemble?)
        
        # Padding
        mask = np.zeros((self.max_tiles, 1))
        if len(features) < self.max_tiles:
            mask[len(features):][:] = np.ones((self.max_tiles-len(features), 1))
            features = np.pad(features, ((0, self.max_tiles-len(features)), (0, 0)), 'constant', constant_values=0)
        
        # Convert to Tensor
        features = torch.from_numpy(features.astype(np.float32))
        mask = torch.from_numpy(mask)
        label = torch.tensor([label])
        
        return features.float(), mask.bool(), label.float()