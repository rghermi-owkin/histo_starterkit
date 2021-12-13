"""Script - Train model."""

# Importations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from histo_starterkit.constants import DF_TRAIN_PATH, WEIGHTS_PATH
from histo_starterkit.utils import fit, save_model, save_pickle, get_model
from histo_starterkit.utils import get_loss, plot_logs_cv
from histo_starterkit.datasets import SlideFeaturesDataset


def main(params):
    # Cross-validation

    # Load data
    df = pd.read_csv(DF_TRAIN_PATH)

    cv_train_loss, cv_val_loss = [], []
    cv_train_metric, cv_val_metric = [], []
    for r in range(params['n_repeats']):
        kf = KFold(n_splits=params['n_splits'], shuffle=True)
        for k, (train_indices, valid_indices) in enumerate(kf.split(df)):
            print('REPEAT:', r, '; SPLIT:', k)

            df_train = df.iloc[train_indices]
            df_valid = df.iloc[valid_indices]

            print('Train set:', len(df_train))
            print('Valid set:', len(df_valid))

            # Datasets
            train_dataset = SlideFeaturesDataset(
                df_train, max_tiles=params['max_tiles'])
            valid_dataset = SlideFeaturesDataset(
                df_valid, max_tiles=params['max_tiles'])

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

            # Training
            model, train_losses, valid_losses, train_metrics, valid_metrics = fit(
                model=model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=params['num_epochs'],
                device=params['device'],
                verbose=False,
            )

            #plot_logs(train_losses, valid_loss, train_metrics, valid_metrics)
            
            print(train_losses[-1], valid_losses[-1])
            print(train_metrics[-1], valid_metrics[-1])
            
            cv_train_loss.append(train_losses[-1])
            cv_val_loss.append(valid_losses[-1])
            cv_train_metric.append(train_metrics[-1])
            cv_val_metric.append(valid_metrics[-1])

    plot_logs_cv(
        cv_train_loss,
        cv_val_loss,
        cv_train_metric,
        cv_val_metric,
    )

    

    # Save model weights and params
    params['save_name']
    save_model(
        os.path.join(WEIGHTS_PATH, '.'.join([params['save_name'], 'pth'])), 
        model,
    )
    save_pickle(
        os.path.join(WEIGHTS_PATH, '.'.join([params['save_name'], 'pkl'])), 
        model,
    )

if __name__ == '__main__':

    # Define parameters
    params = {
        # General params
        'random_state':42,
        'device':'cuda',
        'save_name':'tmp',
        
        # Dataset params
        'max_tiles':100,
        
        # Model params
        'model_name':'MeanPool',
        'loss_name':'BCEWithLogitsLoss',
        
        # Training params
        'batch_size':16,
        'num_epochs':2,
        'learning_rate':1e-4,
        
        # Validation params
        ## Train/valid split
        'valid_size':0.20,
        ## Cross-validation
        'n_repeats':3,
        'n_splits':5,
    }

    main(params)