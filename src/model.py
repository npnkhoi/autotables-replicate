"""
WARNING: non-functional code
"""
# Create a pytorch model to classify tables

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data import data
from sentence_transformers import SentenceTransformer

NAN_STRING = 'N/A'
NUM_CLASSES = 7
EMBEDDING_SIZE = 768
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def force_cudnn_initialization():
    # https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

"""
Embedding layer for pandas dataframe
"""
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        # self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.model = SentenceTransformer('efederici/sentence-bert-base')
    
    """
    encode each cell of a table into a vector, return a tensor of shape (batch_size, num_rows, num_columns, embedding_size)
    """
    def forward(self, x: list[pd.DataFrame]) -> torch.Tensor:
        embeddings = []
        for table in x:
            embeddings.append([])
            for _, row in table.iterrows():
                # change type to string
                row_data = row.apply(str).fillna(NAN_STRING).values
                embeddings[-1].append(self.model.encode(row_data))
        embeddings = np.array(embeddings)
        return torch.tensor(embeddings)
    

"""
Input: a tensor of shape (batch_size, num_rows, num_columns, embedding_size)
Output: a tensor of shape (batch_size, 7)
Steps:
- Dimension reduction layers from embedding_size to 64, then 32
- CNN 1x2 and 1x1 convolution filters
- Fully connected layers followed by softmax
"""
class TableClassifier(nn.Module):
    def __init__(self):
        super(TableClassifier, self).__init__()

        self.fc1 = nn.Linear(EMBEDDING_SIZE, 64)
        self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, NUM_CLASSES)

        self.conv12_col = nn.Conv2d(32, 32, (1, 2))

    def forward(self, x):
        # breakpoint()

        # dimension reduction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        headers = x[:, 0, :, :]
        cols = x[:, 1:, :, :]
        # change shape of rows from (batch_size, num_rows, num_columns, 32) to (batch_size, 32, num_rows, num_columns)
        cols = torch.transpose(cols, 1, 3)
        rows = torch.transpose(cols, 2, 3)
        
        # for columns of shape (num_columns, num_rows, 32), apply 1x2 convolution filter and avg pool to get (num_columns, 32)
        breakpoint()
        tmp = F.relu(self.conv12_col(cols))
        cols = F.avg_pool1d(tmp, cols.size(1))
        cols = torch.squeeze(cols, 1)
        print(cols.shape)
        breakpoint()

        # x = self.fc3(x)
        return F.softmax(x, dim=1)
        

if __name__ == "__main__":
    force_cudnn_initialization()
    print(device)
    table = pd.read_csv('ATBench/stack/stack_test1/data.csv')
    embedding = Embedding()
    x = embedding([table]).to(device)
    print(x.shape, x.dtype, x.device)

    model = TableClassifier()
    model = model.to(device)
    y = model(x)
    print(y.shape, y.dtype, y.device)