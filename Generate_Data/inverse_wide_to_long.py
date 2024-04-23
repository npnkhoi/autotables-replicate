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

def inverse_wide_to_long(df, label_info):
    # Extract the relevant parameters from label_info
    start_idx = label_info['wide_to_long_start_idx']
    end_idx = label_info['wide_to_long_end_idx']
    
    # Columns to pivot back to wide format
    id_vars = df.columns[:start_idx].tolist()
    value_vars = df.columns[start_idx+1:].tolist()
    no_groups = (end_idx - start_idx + 1)//len(value_vars)
    rows = df.values.tolist()
    l = 0
    new_rows = []
    while l < len(rows):
        d = {}
        for k in range(no_groups): 
            row = rows[l]
            for i in range(start_idx):
                d[id_vars[i]] = row[i]
            
            for j in range(len(value_vars)):
                d[str(value_vars[j])+'_'+str(row[start_idx])] = row[start_idx+1+j]
            l+=1
        new_rows.append(d)

    wide_df = pd.DataFrame.from_records(new_rows)
    return wide_df

def augment_dataframe(df, num_copies=1):
    augmented_dfs = []
    num_rows = len(df)
    columns = df.columns.tolist()

    for _ in range(num_copies):
        # Make a copy of the DataFrame to perform operations on
        aug_df = df.copy()

        # Randomly decide which augmentations to apply
        # do_shuffle = np.random.choice([True, False])
        # do_drop_col = np.random.choice([True, False])
        # do_subset_rows = np.random.choice([True, False])

        # if do_shuffle:
        #     aug_df = aug_df.sample(frac=1).reset_index(drop=True)

        # if do_drop_col and len(columns) > 2:  
        #     col_to_drop = np.random.choice(columns[1:],1)
        #     aug_df = aug_df.drop(col_to_drop, axis=1)

        # if do_subset_rows:
        #     subset_fraction = np.random.uniform(0.5, 1)  # Randomly choose to keep between 50% to 100% of the rows
        #     aug_df = aug_df.sample(frac=subset_fraction).reset_index(drop=True)

        augmented_dfs.append(aug_df)

    return augmented_dfs

directory = "../ATBench/wide_to_long/wide_to_long_test"
avail_data = 34
ind = 1
for i in range(1, avail_data+1):
    dataset_dir = directory+str(i)
    gt_df = pd.read_csv(os.path.join(dataset_dir, "gt.csv"))
    label_info = read_label_info(os.path.join(dataset_dir, 'info.json'))
    dfs = augment_dataframe(gt_df)

    train = i <= avail_data*0.8
    for df in dfs:
        inverse_df = inverse_wide_to_long(df, label_info)
        if train:
            path = f"../Data/wide_to_long/train/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = f"../Data/wide_to_long/test/{ind}"
            if not os.path.exists(path):
                os.makedirs(path)
        df.to_csv(os.path.join(path, "gt.csv"), index=False)
        inverse_df.to_csv(os.path.join(path, "data.csv"), index=False)
        ind+=1
