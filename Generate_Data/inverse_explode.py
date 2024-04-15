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

def inverse_explode(df, explode_column_name):

    df.fillna('NaN', inplace=True)
    # Group by all other columns except the explode column and aggregate the explode column
    group_columns = [col for col in df.columns if col != explode_column_name]
    agg_df = df.groupby(group_columns)[explode_column_name].agg(lambda x: ', '.join(x.astype(str))).reset_index()

    return agg_df

def augment_dataframe(df, label_info, num_copies=10):
    augmented_dfs = []
    num_rows = len(df)
    columns = df.columns.tolist()
    explode_column_idx = label_info["explode_column_idx"]
    explode_column_name = df.columns[explode_column_idx]

    for _ in range(num_copies):
        # Make a copy of the DataFrame to perform operations on
        aug_df = df.copy()
        flag = False
        # Randomly decide which augmentations to apply
        do_shuffle = np.random.choice([True, False])
        do_drop_col = np.random.choice([True, False])
        do_subset_rows = np.random.choice([True, False])

        if do_shuffle:
            aug_df = aug_df.sample(frac=1).reset_index(drop=True)

        if do_drop_col and len(columns) > 2:  
            col_to_drop = np.random.choice(columns,1)
            if col_to_drop != explode_column_name:
                aug_df = aug_df.drop(col_to_drop, axis=1)

        if do_subset_rows:
            subset_fraction = np.random.uniform(0.5, 1)  # Randomly choose to keep between 50% to 100% of the rows
            aug_df = aug_df.sample(frac=subset_fraction).reset_index(drop=True)

        augmented_dfs.append(aug_df)

    return augmented_dfs, explode_column_name

directory = "../ATBench/explode/explode_test"
avail_data = 48
ind = 1
for i in range(1, avail_data+1):
    dataset_dir = directory+str(i)
    gt_df = pd.read_csv(os.path.join(dataset_dir, "gt.csv"))
    label_info = read_label_info(os.path.join(dataset_dir, 'info.json'))
    dfs, explode_column_name = augment_dataframe(gt_df, label_info)

    train = i <= avail_data*0.8
    for df in dfs:
        inverse_df = inverse_explode(df, explode_column_name)
        if train:
            path = f"../Data/explode/train/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = f"../Data/explode/test/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        df.to_csv(os.path.join(path, "gt.csv"), index=False)
        inverse_df.to_csv(os.path.join(path, "data.csv"), index=False)
        ind+=1
