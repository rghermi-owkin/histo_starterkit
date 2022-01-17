"""Script - Extract and save masks, features and metadata for each slide."""

# Importations
import os
import pandas as pd
from tqdm import tqdm

from tilingtool.core import histo
from tilingtool.filters.matter_detection import BUNet
from tilingtool.extractors import ResNet50Keras, WideResNetCOAD

from histo_starterkit.constants import DF_PATH, SLIDE_PATH
from histo_starterkit.constants import MASK_PATH, FEATURES_PATH, METADATA_PATH
from histo_starterkit.constants import IMAGENET_FEATURES_PATH, MOCO_COAD_FEATURES_PATH
from histo_starterkit.utils import load_slide, load_mask
from histo_starterkit.utils import save_mask, save_features, save_metadata


def get_model(features_name):
    if features_name == 'ImageNet':
        return ResNet50Keras()
    if features_name == 'MoCo-COAD':
        return WideResNetCOAD()

# Get features path
if params['features_name'] == 'ImageNet':
    FEATURES_PATH = IMAGENET_FEATURES_PATH
elif params['features_name'] == 'MoCo-COAD':
    FEATURES_PATH = MOCO_COAD_FEATURES_PATH


def main(params):

    # Import data
    df = pd.read_csv(DF_PATH)

    # Prepare models
    if params['extract_mask']:
        matter_detector = BUNet()
    extractor = get_model(params['features_name'])

    # Extraction
    for i in tqdm(range(len(df)), total=len(df)):
        row = df.iloc[i]
        slide_name = str(row.slide_name)

        try:
            filename = '.'.join([slide_name, params['slide_extension']])
            slide_path = os.path.join(SLIDE_PATH, filename)
            slide = load_slide(slide_path)

            if params['extract_mask']:
                mask, features, metadata = histo.extract(
                    slide,
                    matter_detector,
                    extractor,
                )

                save_mask(MASK_PATH, slide_name, mask)
                save_features(FEATURES_PATH, slide_name, features)
                save_metadata(METADATA_PATH, slide_name, metadata)
            else:
                mask = load_mask(MASK_PATH, slide_name)

                features = histo.extract_with_mask(
                    slide=slide,
                    mask=mask,
                    feature_extractor=extractor,
                )

                save_features(FEATURES_PATH, slide_name, features)
            
        except:
            print('Error with slide:', slide_name)

if __name__ == '__main__':
    params = {
        'extract_mask':False,
        'features_name':'ImageNet', # ImageNet, MoCo-COAD
        'slide_extension':'tif',
    }
    main(params)
