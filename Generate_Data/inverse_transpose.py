import pandas as pd
import numpy as np
import json
import os

def augment_dataframe(df, num_copies=10):
    augmented_dfs = []
    num_rows = len(df)
    columns = df.columns.tolist()

    for _ in range(num_copies):
        # Make a copy of the DataFrame to perform operations on
        aug_df = df.copy()

        # Randomly decide which augmentations to apply
        do_shuffle = np.random.choice([True, False])
        do_drop_col = np.random.choice([True, False])
        do_subset_rows = np.random.choice([True, False])
        do_shuffle_cols= np.random.choice([True, False])

        if do_shuffle:
            aug_df = aug_df.sample(frac=1).reset_index(drop=True)

        if do_drop_col and len(columns) > 2:  # Ensure there's at least one column to drop
            col_to_drop = np.random.choice(columns[1:],1)
            aug_df = aug_df.drop(col_to_drop, axis=1)

        if do_subset_rows:
            subset_fraction = np.random.uniform(0.5, 1)  # Randomly choose to keep between 50% to 100% of the rows
            aug_df = aug_df.sample(frac=subset_fraction).reset_index(drop=True)

        if do_shuffle_cols:
            shuffled_columns = np.random.permutation(aug_df.columns)
            aug_df = aug_df[shuffled_columns]

        augmented_dfs.append(aug_df)

    return augmented_dfs

def inverse_transpose(df):
    return df.set_index(df.columns.to_list()).T

directory = "../ATBench/transpose/transpose_test"
avail_data = 28
ind = 1
for i in range(1, avail_data+1):
    dataset_dir = directory+str(i)
    gt_df = pd.read_csv(os.path.join(dataset_dir, "gt.csv"))

    dfs = augment_dataframe(gt_df)

    train = i <= avail_data*0.8
    for df in dfs:
        inverse_df = inverse_transpose(df)
        if train:
            path = f"../Data/transpose/train/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = f"../Data/transpose/test/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        df.to_csv(os.path.join(path, "gt.csv"), index=False)
        inverse_df.to_csv(os.path.join(path, "data.csv"), index=False)
        ind+=1
