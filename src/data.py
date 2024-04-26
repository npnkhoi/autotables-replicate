"""
list the path of all the testcases
"""

import pandas as pd
import os
import json
from torch.utils.data import Dataset

CLASSES = ['explode', 'ffill', 'pivot', 'stack', 'subtitle', 'transpose', 'wide_to_long']
operator2idx = {op: i for i, op in enumerate(CLASSES)}

def get_test_data():
    # Load test data
    base_path = 'ATBench'
    test_data = []
    for root, dirs, files in os.walk(base_path):
        ancestors = root.split('/')
        if len(ancestors) != 2:
            continue
        if ancestors[-1] == 'multistep':
            continue

        for dir in dirs:
            # print(dir)

            # one testcase
            info = json.load(open(os.path.join(root, dir, 'info.json')))
            assert len(info['label']) == 1
            operator = info['label'][0]['operator']
            assert operator == ancestors[-1]

            input_path = os.path.join(root, dir, 'data.csv')
            test_data.append((input_path, operator2idx[operator]))
    
    return test_data

def get_train_val_data():
    # Load train & val data
    base_path = 'Data'
    train_data = []
    val_data = []

    for root, dirs, files in os.walk(base_path):
        ancestors = root.split('/')
        if len(ancestors) != 4 or 'data.csv' not in files:
            continue
        
        operator = ancestors[1]
        input_path = os.path.join(root, 'data.csv')
        split = ancestors[2]
        if split == 'train':
            train_data.append((input_path, operator2idx[operator]))
        elif split == 'test':
            val_data.append((input_path, operator2idx[operator]))
        else:
            raise ValueError(f'Invalid split: {split}')
    
    return train_data, val_data

class TableDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, operator = self.data[idx]
        # table = pd.read_csv(path)
        return path, operator


if __name__ == "__main__":
    test = get_test_data()
    dataset = TableDataset(test)
    print(len(dataset))
    print(dataset[0])