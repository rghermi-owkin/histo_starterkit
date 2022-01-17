import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import openslide
import torch
from torch.nn import BCEWithLogitsLoss

from classic_algos.nn import CoxLoss, SmoothCindexLoss

from histo_starterkit.models import MeanPool, MaxPool, Chowder, DeepMIL


# Loading

def load_array(path: str) -> np.ndarray:
    # Load a .npy file.
    return np.load(path)

def load_image(path: str) -> np.ndarray:
    # Load a .png image as a numpy array.
    return np.array(Image.open(path).getdata())

def load_pickle(path: str):
    # Load a .pkl file.
    return pickle.load(open(path, 'rb'))

def load_model(path: str, model: torch.nn.Module) -> torch.nn.Module:
    # Load a .pt or .pth state_dict file.
    return model.load_state_dict(torch.load(path))

def load_slide(slide_path: str) -> openslide.OpenSlide:
    # Load a histology slide (.svs, .tif, ...) with OpenSlide.
    return openslide.open_slide(slide_path)

def load_mask(mask_path: str, slide_name: str) -> None:
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(mask_path, filename)
    return load_array(filepath)

def load_features(features_path: str, slide_name: str) -> None:
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(features_path, filename)
    return load_array(filepath)

def load_metadata(metadata_path: str, slide_name: str) -> None:
    filename = '.'.join([slide_name, 'pkl'])
    filepath = os.path.join(metadata_path, filename)
    return load_pickle(filepath)


# Saving

def save_array(path: str, array: np.ndarray) -> None:
    # Save a numpy array as a .npy file.
    np.save(path, array)

def save_image(path: str, array: np.ndarray) -> None:
    # Save a numpy array as a .png image.
    Image.fromarray(array).save(path)

def save_pickle(path: str, dictionary) -> None:
    # Save a dictionary as a .pkl file.
    with open(path, 'wb') as file:
        pickle.dump(dictionary, file, pickle.HIGHEST_PROTOCOL)

def save_model(path: str, model: torch.nn.Module) -> None:
    # Save a PyTorch model as a .pt or .pth state_dict file.
    torch.save(model.state_dict(), path)

def save_mask(mask_path: str, slide_name: str, mask: np.ndarray) -> None:
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(mask_path, filename)
    save_array(filepath, features)

def save_features(features_path: str, slide_name: str, features: np.ndarray) -> None:
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(features_path, filename)
    save_array(filepath, features)

def save_metadata(metadata_path: str, slide_name: str, metadata) -> None:
    filename = '.'.join([slide_name, 'pkl'])
    filepath = os.path.join(metadata_path, filename)
    save_pickle(filepath, metadata)


# Getting

def get_model(model_name: str, in_features: int, out_features: int):
    if model_name == 'MeanPool':
        return MaxPool(in_features=in_features, out_features=out_features)
    elif model_name == 'MaxPool':
        return MeanPool(in_features=in_features, out_features=out_features)
    elif model_name == 'Chowder':
        return Chowder.galaxy_survival(in_features=in_features, out_features=out_features)
    elif model_name == 'DeepMIL':
        return DeepMIL.survival(in_features=in_features, out_features=out_features)
        
def get_loss(loss_name: str):
    # Classification losses
    if loss_name == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss()
    # Survival losses
    if loss_name == 'CoxLoss':
        return CoxLoss()
    if loss_name == 'SmoothCindexLoss':
        return SmoothCindexLoss()