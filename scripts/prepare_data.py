"""Script - Prepare and save dataframe."""

import os
import pandas as pd

from histo_starterkit.constants import DF_TRAIN_PATH, DF_TEST_PATH, DF_PATH

def main():

    df_train = pd.read_csv(
        '/STORAGE/data/Camelyon_clean/train_data_index.csv')
    df_train = df_train.rename(columns={'Unnamed: 0':'slide_name'})
    df_train['target'] = df_train.target.apply(lambda x:1 if x == 'Tumor' 
                                                          else 0)

    df_test = pd.read_csv(
        '/STORAGE/data/Camelyon_clean/test_data_index.csv')
    df_test = df_test.rename(columns={'Unnamed: 0':'slide_name'})
    df_test['target'] = df_test.target.apply(lambda x:1 if x == 'Tumor' 
                                                        else 0)

    df_train.to_csv(DF_TRAIN_PATH, index=False)
    df_test.to_csv(DF_TEST_PATH, index=False)

    df_train['train'] = [1]*len(df_train)
    df_test['train'] = [0]*len(df_test)
    df = pd.concat([df_train, df_test])

    df.to_csv(DF_PATH, index=False)

if __name__ == '__main__':
    main()