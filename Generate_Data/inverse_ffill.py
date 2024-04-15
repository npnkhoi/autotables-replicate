import pandas as pd
import numpy as np
import json
import os

def read_label_info(json_file_path):
    # Open and read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    # Extract the label information
    label_info = data.get('label', [])[0]  # Assuming there's at least one label and taking the first
    
    return label_info

def inverse_ffill(df, label_info):
    ffill_end_idx = label_info['ffill_end_idx']
    columns = df.columns.to_list()

    rows = df.values.tolist()
    data = []
    for i in range(ffill_end_idx+1):
        prev = rows[0][i]
        new_column = [prev]
        for j in range(1, len(rows)):
            if rows[j][i] == prev:
                new_column.append(np.nan)
            else:
                prev = rows[j][i]
                new_column.append(prev)
        data.append(new_column)
    for i in range(ffill_end_idx+1, len(df.columns)):
        new_column = []
        for j in range(len(rows)):
            new_column.append(rows[j][i])
        data.append(new_column)
    df = pd.DataFrame(data).T
    df.columns = columns
    return df

def augment_dataframe(df, label_info, num_copies=10):
    augmented_dfs = []
    num_rows = len(df)
    columns = df.columns.tolist()
    ffill_end_idx = label_info['ffill_end_idx']

    for _ in range(num_copies):
        # Make a copy of the DataFrame to perform operations on
        aug_df = df.copy()

        # Randomly decide which augmentations to apply
        do_shuffle = np.random.choice([True, False])
        do_drop_col = np.random.choice([True, False])
        do_subset_rows = np.random.choice([True, False])

        if do_shuffle:
            aug_df = aug_df.sample(frac=1).reset_index(drop=True)

        if do_drop_col and len(columns) > ffill_end_idx+1:  # Ensure there's at least one column to drop
            col_to_drop = np.random.choice(columns[ffill_end_idx+1:],1)
            aug_df = aug_df.drop(col_to_drop, axis=1)

        if do_subset_rows:
            subset_fraction = np.random.uniform(0.5, 1)  # Randomly choose to keep between 50% to 100% of the rows
            aug_df = aug_df.sample(frac=subset_fraction).reset_index(drop=True)

        augmented_dfs.append(aug_df)

    return augmented_dfs

directory = "../ATBench/ffill/ffill_test"
avail_data = 18
ind = 1
for i in range(1, avail_data+1):
    dataset_dir = directory+str(i)
    gt_df = pd.read_csv(os.path.join(dataset_dir, "gt.csv"))
    label_info = read_label_info(os.path.join(dataset_dir, 'info.json'))
    dfs = augment_dataframe(gt_df, label_info)

    train = i <= avail_data*0.8
    for df in dfs:
        inverse_df = inverse_ffill(df, label_info)
        if train:
            path = f"../Data/ffill/train/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = f"../Data/ffill/test/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        df.to_csv(os.path.join(path, "gt.csv"), index=False)
        inverse_df.to_csv(os.path.join(path, "data.csv"), index=False)
        ind+=1
