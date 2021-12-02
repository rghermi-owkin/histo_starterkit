"""Script - Extract and save masks and features for each slide."""

# Importations
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from tilingtool.core import histo
from tilingtool.filters.matter_detection import BUNet
from tilingtool.extractors import ResNet50Keras

from histo_starterkit import load_slide

# Constants
DATA_PATH = '/STORAGE/data/Camelyon_clean/slides'
SAVE_PATH = './data/'

# Create dataframe
filenames = os.listdir(DATA_PATH)

df = pd.DataFrame(data=filenames, columns=['filename'])
df['label'] = df.filename.apply(lambda x:1 if x.split('_')[0] == 'Tumor' else 0)
df['test'] = df.filename.apply(lambda x:1 if x.split('_')[0] == 'Test' else 0)

# Extraction
matter_detector = BUNet()
extractor = ResNet50Keras()

for i, row in tqdm(df[250:].iterrows(), total=len(df[250:])):
    filename = row.filename
    
    slide_name = filename
    slide_path = os.path.join(DATA_PATH, slide_name)
    slide = load_slide(slide_path)

    mask, features, metadata = histo.extract(slide,
                                             matter_detector,
                                             extractor)
    
    name = filename.split('.')[0]
    
    mask_name = '_'.join([name, 'mask'])
    mask_name = '.'.join([mask_name, 'npy'])
    mask_path = os.path.join(SAVE_PATH, mask_name)
    
    features_name = '_'.join([name, 'features'])
    features_name = '.'.join([features_name, 'npy'])
    features_path = os.path.join(SAVE_PATH, features_name)

    metadata_name = '_'.join([name, 'metadata'])
    metadata_name = '.'.join([metadata_name, 'pkl'])
    metadata_path = os.path.join(SAVE_PATH, metadata_name)
    
    np.save(mask_path, mask)
    np.save(features_path, features)

    with open(metadata_path, 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file, pickle.HIGHEST_PROTOCOL)