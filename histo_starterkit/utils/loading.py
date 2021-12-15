import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import openslide
import torch
from torch.nn import BCEWithLogitsLoss

from histo_starterkit.constants import MASK_PATH, FEATURES_PATH
from histo_starterkit.constants import METADATA_PATH


# Loading and saving

def load_array(path: str):
    # Load a .npy file.
    return np.load(path)

def save_array(path: str, array: np.ndarray):
    # Save a numpy array as a .npy file.
    np.save(path, array)

def load_image(path: str):
    # Load a .png image as a numpy array.
    return np.array(Image.open(path).getdata())

def save_image(path: str, array: np.ndarray):
    # Save a numpy array as a .png image.
    Image.fromarray(array).save(path)

def load_pickle(path: str):
    # Load a .pkl file.
    return pickle.load(open(path, 'rb'))

def save_pickle(path, dictionary):
    with open(path, 'wb') as file:
        pickle.dump(dictionary, file, pickle.HIGHEST_PROTOCOL)

def load_model(path: str, model: torch.nn.Module):
    model.load_state_dict(torch.load(path))

def save_model(path: str, model: torch.nn.Module):
    # Save a PyTorch model as a .pt or .pth file.
    torch.save(model.state_dict(), path)

def load_slide(slide_path: str):
    return openslide.open_slide(slide_path)

def load_mask(slide_name: str):
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(MASK_PATH, filename)
    return load_array(filepath)

def save_mask(slide_name: str, mask):
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(MASK_PATH, filename)
    save_array(filepath, mask)

def load_features(slide_name: str):
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(FEATURES_PATH, filename)
    return load_array(filepath)

def save_features(slide_name: str, features):
    filename = '.'.join([slide_name, 'npy'])
    filepath = os.path.join(FEATURES_PATH, filename)
    save_array(filepath, features)

def load_metadata(slide_name: str):
    filename = '.'.join([slide_name, 'pkl'])
    filepath = os.path.join(METADATA_PATH, filename)
    return load_pickle(filepath)

def save_metadata(slide_name: str, metadata):
    filename = '.'.join([slide_name, 'pkl'])
    filepath = os.path.join(METADATA_PATH, filename)
    save_pickle(filepath, metadata)
    
def get_loss(loss_name: str):
    if loss_name == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss()