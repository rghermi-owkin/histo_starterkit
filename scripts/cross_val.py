"""Script - Train model (cross-validation)."""

# Importations
import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from histo_starterkit.constants import DF_PATIENT_PATH, DF_SLIDE_PATH, DF_CROSS_VAL
from histo_starterkit.constants import IMAGENET_FEATURES_PATH, MOCO_COAD_FEATURES_PATH
from histo_starterkit.utils import load_data, training


def main(params):
    # Cross-validation split
    print('Cross-validation')

    # Get features path
    if params['features_name'] == 'ImageNet':
        FEATURES_PATH = IMAGENET_FEATURES_PATH
    elif params['features_name'] == 'MoCo-COAD':
        FEATURES_PATH = MOCO_COAD_FEATURES_PATH

    # Load data
    df = pd.read_csv(DF_CROSS_VAL)


    # R repeats, S splits
    cv_train_loss, cv_val_loss = [], []
    cv_train_metric, cv_val_metric = [], []
    for r in range(params['n_repeats']):
        for s in range(params['n_splits']):

            print('REPEAT:', r, '; SPLIT:', s)
            
            col = '_'.join(['REPEAT', str(r), 'SPLIT', str(s)])
            df_train = df[col == 1]
            df_valid = df[col == 0]

            print('Train set:', len(df_train), '; Valid set:', len(df_valid))
            
            train_losses, valid_losses, train_metrics, valid_metrics = training(
                df_train=df_train, 
                df_valid=df_valid, 
                params=params, 
                FEATURES_PATH=FEATURES_PATH, 
                log_metrics=True,
            )

            train_loss, valid_loss = train_losses[-1], valid_losses[-1]
            train_metric, valid_metric = train_metrics[-1], valid_metrics[-1]

            cv_train_loss.append(train_loss)
            cv_val_loss.append(valid_loss)
            cv_train_metric.append(train_metric)
            cv_val_metric.append(valid_metric)

            logs = {
                'cv_train_loss':train_loss,
                'cv_val_loss':valid_loss,
                'cv_train_metric':train_metric,
                'cv_val_metric':valid_metric,
            }
            mlflow.log_metrics(logs, step=int(r+s))

            print('Train loss:', train_loss, '; Valid loss:', valid_loss)
            print('Train metric:', train_metric, '; Valid metric:', valid_metric)

    print('Median C-Index (train): {:.2f} ({:.2f})'.format(
        np.median(cv_train_metric), np.std(cv_train_metric)))
    print('Median C-Index (valid): {:.2f} ({:.2f})'.format(
        np.median(cv_val_metric), np.std(cv_val_metric)))


if __name__ == '__main__':

    # Define parameters
    params = {
        # General params
        'random_state':42,
        'device':'cuda',
        'verbose':False,

        # Dataset params
        'in_features':2048,
        'out_features':1,
        'dimensionality-reduction':True,
        'feature_dim':256,

        # Model params
        'features_name':'MoCo-COAD',     # ImageNet, MoCo-COAD
        'model_name':'MaxPool',          # MeanPool, MaxPool, Chowder, DeepMIL
        'loss_name':'SmoothCindexLoss',  # CoxLoss, SmoothCindexLoss
        'max_tiles':10_000,

        # Training params
        'batch_size':8,
        'num_epochs':30,
        'learning_rate':1e-4,

        # Validation params
        ## Train/valid split
        'valid_size':0.20,
        ## Cross-validation
        'n_repeats':10,
        'n_splits':5,
    }

    run_name = '_'.join([
        'cross-val',
        'isDimensionalityReduction', 
        str(params['dimensionality-reduction']), 
        'featureDim', str(params['feature_dim']), 
        str(params['features_name']), 
        str(params['model_name']), 
        str(params['loss_name']),
        str(params['max_tiles']),
        str(params['batch_size']),
        str(params['num_epochs']),
        str(params['learning_rate']),
    ])
    print(run_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        main(params)

    #plot_logs_cv(
    #    cv_train_loss,
    #    cv_val_loss,
    #    cv_train_metric,
    #    cv_val_metric,
    #)
