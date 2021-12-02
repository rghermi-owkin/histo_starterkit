import numpy as np

import torch
from torch.utils.data import Dataset

from histo_starterkit import *

class CamelyonDataset(Dataset):
    
    def __init__(self, df, max_tiles=1000, feature_dim=2048, save_path='../data'):
        self.df = df
        self.max_tiles = max_tiles
        self.feature_dim = feature_dim
        self.save_path = save_path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get label
        label = row.target
        
        # Get features
        slide_name = row.slide_name
        features = load_features(slide_name, self.save_path)
        metadata= load_metadata(slide_name, self.save_path)
        
        # Sub-sample features
        ## Fixed random sub-sampling
        features = features[:self.max_tiles, 3:] # keep coordinates and discard them in the model?
        ## Variable random sub-sampling
        ## sort of data augmentation (different tiles selected for the same slide)
        ## need to handle inference time (fixed sampling? ensemble?)
        
        # Padding
        if len(features) < self.max_tiles:
            tmp = np.zeros((self.max_tiles, self.feature_dim))
            tmp[:len(features), :] = features
            features = tmp
        ## need to return padding indices, useful for the following
        
        # Convert to Tensor
        features = torch.from_numpy(features.astype(np.float32)).float()
        label = torch.tensor([label]).float()
        
        return features, label