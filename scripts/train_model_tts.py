"""Script - Train model (train/test split)."""

# Importations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from histo_starterkit.constants import DF_TRAIN_PATH, DF_TEST_PATH, WEIGHTS_PATH, MLFLOW_PATH
from histo_starterkit.utils import fit, save_model, save_pickle
from histo_starterkit.utils import get_loss, plot_logs_tts
from histo_starterkit.datasets import SlideFeaturesDataset

from histo_starterkit.models import MeanPool, MaxPool, CustomDeepMIL
from classic_algos.nn import Chowder, DeepMIL, MultiHeadDeepMIL

import mlflow


def get_model(model_name):
    if model_name == 'DeepMIL':
        return DeepMIL.classification(in_features=2048, out_features=1)
    elif model_name == 'Chowder':
        return Chowder.classification(in_features=2048, out_features=1)
    elif model_name == 'MultiHeadDeepMIL':
        return MultiHeadDeepMIL.classification(in_features=2048, out_features=1)
    elif model_name == 'MeanPool':
        return MeanPool(in_features=2048, out_features=1)
    elif model_name == 'MaxPool':
        return MaxPool(in_features=2048, out_features=1)
    elif model_name == 'CustomDeepMIL':
        return CustomDeepMIL(in_features=2048, out_features=1)


def main(params):
    # Train/valid split
    print('Train/test split')

    # Load data
    df_train = pd.read_csv(DF_TRAIN_PATH)
    df_valid = pd.read_csv(DF_TEST_PATH)

    # Train/valid split
    #df_train, df_valid = train_test_split(df, test_size=params['valid_size'], random_state=params['random_state'])

    print('Train set:', len(df_train))
    print('Valid set:', len(df_valid))

    # Datasets
    train_dataset = SlideFeaturesDataset(df_train, max_tiles=params['max_tiles'])
    valid_dataset = SlideFeaturesDataset(df_valid, max_tiles=params['max_tiles'])

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
    model = get_model(params['model_name']).to(params['device'])
    criterion = get_loss(params['loss_name']).to(params['device'])
    optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    
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
        verbose=True,
    )

    #plot_logs_tts(train_losses, valid_losses, train_metrics, valid_metrics)
    
    # Save model weights and params
    #save_model(
    #    os.path.join(WEIGHTS_PATH, '.'.join([params['save_name'], 'pth'])), 
    #    model,
    #)
    #save_pickle(
    #    os.path.join(WEIGHTS_PATH, '.'.join([params['save_name'], 'pkl'])), 
    #    model,
    #)

if __name__ == '__main__':

    mlflow.set_tracking_uri(MLFLOW_PATH)
    mlflow.set_experiment('model_name')

    with mlflow.start_run():

        # Define parameters
        params = {
            # General params
            'random_state':42,
            'device':'cuda:1',

            # Dataset params
            'max_tiles':10_000,

            # Model params
            'model_name':'Chowder',
            'loss_name':'BCEWithLogitsLoss',

            # Training params
            'batch_size':32,
            'num_epochs':20,
            'learning_rate':1e-3,

            # Validation params
            ## Train/valid split
            'valid_size':0.20,
            ## Cross-validation
            'n_repeats':3,
            'n_splits':5,
        }
        mlflow.log_params(params)
        
        main(params)