import numpy as np

import torch
from torch.utils.data import Dataset

from histo_starterkit.utils import load_features


class ClassificationDataset(Dataset):
    
    def __init__(
        self, 
        df, 
        features_path, 
        max_tiles=1000, 
        feature_dim=2048,
        transform=None,
    ):
        self.df = df
        self.features_path = features_path
        self.max_tiles = max_tiles
        self.feature_dim = feature_dim
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get label
        label = row.target
        
        # Get features
        slide_name = str(row.slide_name)
        features = load_features(self.features_path, slide_name)
        
        # Sub-sample features
        ## Fixed random sub-sampling
        features = features[:self.max_tiles, 3:]
        ## Variable random subsampling
        #np.random.shuffle(features)
        #features = features[:self.max_tiles, 3:]
        
        # Padding
        mask = np.zeros((self.max_tiles, 1))
        if len(features) < self.max_tiles:
            mask[len(features):][:] = np.ones((self.max_tiles-len(features), 1))
            features = np.pad(
                features, 
                ((0, self.max_tiles-len(features)), (0, 0)), 
                'constant', 
                constant_values=0,
            )
        
        # Convert to Tensor
        features = torch.from_numpy(features.astype(np.float32))
        mask = torch.from_numpy(mask)
        label = torch.tensor([label])

        # Transforms
        if self.transform is not None:
            features = self.transform(features)
        
        return features.float(), mask.bool(), label.float()


class SurvivalDataset(Dataset):
    
    def __init__(
        self, 
        df, 
        features_path, 
        max_tiles=1000, 
        feature_dim=2048,
        transform=None,
    ):
        self.df = df
        self.features_path = features_path
        self.max_tiles = max_tiles
        self.feature_dim = feature_dim
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get label
        label = row.PFS_time if row.PFS_event == 1 else - row.PFS_time
        
        # Get features
        slide_name = str(row.slide_name)
        features = load_features(self.features_path, slide_name)
        
        # Sub-sample features
        ## Fixed random sub-sampling
        features = features[:self.max_tiles, 3:]
        ## Variable random subsampling
        #np.random.shuffle(features)
        #features = features[:self.max_tiles, 3:]
        
        # Padding
        mask = np.zeros((self.max_tiles, 1))
        if len(features) < self.max_tiles:
            mask[len(features):][:] = np.ones((self.max_tiles-len(features), 1))
            features = np.pad(
                features, 
                ((0, self.max_tiles-len(features)), (0, 0)), 
                'constant', 
                constant_values=0,
            )
        
        # Convert to Tensor
        features = torch.from_numpy(features.astype(np.float32))
        mask = torch.from_numpy(mask)
        label = torch.tensor([label])

        # Transforms
        if self.transform is not None:
            features = self.transform(features)
        
        return features.float(), mask.bool(), label.float()