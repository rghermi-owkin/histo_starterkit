# Importations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam

from classic_algos.utils.data.slides import fit_auto_encoder
from classic_algos.transforms import Encode
from classic_algos.nn import AutoEncoder

from histo_starterkit.utils import get_model, get_loss, fit
from histo_starterkit.dataloaders import SurvivalDataset


def load_data(DF_PATH, FEATURES_PATH, truncated=False):
    # Load data
    df = pd.read_csv(DF_PATH)

    ## Only keep patients for which we have at least one feature matrix
    available_ids = [int(x.split('.')[0]) for x in os.listdir(FEATURES_PATH)]
    df = df[(df.slide_name.isin(available_ids))]

    ## Subsample
    #df = df.sample(n=20)

    return df

def training(
    df_train, 
    df_valid, 
    params, 
    FEATURES_PATH, 
    log_metrics=True,
    ):

    if params['dimensionality-reduction']:
        # Dimensionality reduction
        auto_encoder = AutoEncoder(
            in_features=params['in_features'], hidden=[params['feature_dim']], bias=False)

        train_features_paths = [str(slide_name) for slide_name in df_train.slide_name.values]
        train_features_paths = ['.'.join([slide_name, 'npy']) for slide_name in train_features_paths]
        train_features_paths = [os.path.join(FEATURES_PATH, filename) for filename in train_features_paths]
        train_features_paths = [Path(filepath) for filepath in train_features_paths]

        fit_auto_encoder(
            train_features_paths,
            auto_encoder=auto_encoder,
            tiling_tool_format=True,
            max_tiles_seen=100_000,
        )

        transform = Encode(auto_encoder.encoder)
    else:
        transform = None

    # Datasets
    train_dataset = SurvivalDataset(
        df_train,
        FEATURES_PATH,
        max_tiles=params['max_tiles'],
        transform=transform,
    )
    valid_dataset = SurvivalDataset(
        df_valid,
        FEATURES_PATH,
        max_tiles=params['max_tiles'],
        transform=transform,
    )

    # Dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        shuffle=True, 
        batch_size=params['batch_size'],
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, 
        shuffle=False, 
        batch_size=params['batch_size'],
    )

    # Define model, loss, optimizer
    if params['dimensionality-reduction']:
        model = get_model(
            params['model_name'],
            params['in_features'],
            params['out_features'],
        ).to(params['device'])
    else:
        model = get_model(
            params['model_name'],
            params['feature_dim'],
            params['out_features'],
        ).to(params['device'])

    criterion = get_loss(
        params['loss_name'], 
    ).to(params['device'])

    optimizer = Adam(
        model.parameters(), 
        lr=params['learning_rate'],
    )
    
    print('Model:', params['model_name'])
    print('Loss:', params['loss_name'])

    # Training
    train_losses, valid_losses, train_metrics, valid_metrics = fit(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=params['num_epochs'],
        device=params['device'],
        verbose=params['verbose'],
        log_metrics=log_metrics,
    )

    return train_losses, valid_losses, train_metrics, valid_metrics