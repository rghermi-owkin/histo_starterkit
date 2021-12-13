"""Script - Extract and save masks, features and metadata for each slide."""

# Importations
import os
import pandas as pd
from tqdm import tqdm

from tilingtool.core import histo
from tilingtool.filters.matter_detection import BUNet
from tilingtool.extractors import ResNet50Keras

from histo_starterkit.constants import DF_PATH, SLIDE_PATH
from histo_starterkit.utils import load_slide
from histo_starterkit.utils import save_mask, save_features, save_metadata

def main():

    # Import data
    df = pd.read_csv(DF_PATH)

    # Prepare models
    matter_detector = BUNet()
    extractor = ResNet50Keras()

    # Extraction
    for i in tqdm(range(len(df)), total=len(df)):
        row = df.iloc[i]
        slide_name =row.slide_name

        try:
            filename = '.'.join([slide_name, 'tif'])
            slide_path = os.path.join(SLIDE_PATH, filename)
            slide = load_slide(slide_path)
            
            mask, features, metadata = histo.extract(
                slide,
                matter_detector,
                extractor,
            )

            save_mask(slide_name, mask)
            save_features(slide_name, features)
            save_metadata(slide_name, metadata)
        except:
            print('Error with slide:', slide_name)

if __name__ == '__main__':
    main()