"""
WARNING: non-functional code
"""
# Create a pytorch model to classify tables

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.data import test_data
from sentence_transformers import SentenceTransformer

"""
Original dimensions from the paper:
NUM_ROWS = 100 # excluding 1 header
NUM_COLS = 50
EMBEDDING_SIZE = 384
REDUCED_EMBED_SIZE = 32
FINAL_CNN_DIM = 4
FFNN_INTER_DIM = 64
"""

NUM_ROWS = 32 # excluding 1 header
NUM_COLS = 16
EMBEDDING_SIZE = 384
REDUCED_EMBED_SIZE = 32
FINAL_CNN_DIM = 2
FFNN_INTER_DIM = 32

NUM_CLASSES = 8
NAN_STRING = 'N/A'

"""
Embedding layer for pandas dataframe
"""
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        # self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        # self.model = SentenceTransformer('efederici/sentence-bert-base')
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')
    """
    encode each cell of a table into a vector, return a tensor of shape (batch_size, num_rows, num_columns, embedding_size)
    """
    def forward(self, x: list[pd.DataFrame]) -> torch.Tensor:
        batch_embeddings = []
        for table in x:
            table_embedding = np.zeros((NUM_ROWS + 1, NUM_COLS, EMBEDDING_SIZE))
            get_row_embedding = lambda row: self.model.encode(row.apply(str).fillna(NAN_STRING).values)[:NUM_COLS, :]
            
            # header
            header_embedding = get_row_embedding(pd.Series(table.columns))
            table_embedding[0, :header_embedding.shape[0], :] = header_embedding
            
            # rows
            for i_row, row in table.iterrows():
                # change type to string
                row_embedding = get_row_embedding(row)
                table_embedding[i_row, :row_embedding.shape[0], :] = row_embedding

                if i_row >= min(NUM_ROWS, table.shape[0]):
                    break

            batch_embeddings.append(table_embedding)
        batch_embeddings = np.array(batch_embeddings)
        return torch.tensor(batch_embeddings)
    

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

        # TODO: take the dimensions as parameters
        self.dimrec_conv1 = nn.Conv2d(EMBEDDING_SIZE, 64, 1)
        self.dimrec_conv2 = nn.Conv2d(64, REDUCED_EMBED_SIZE, 1)

        self.conv1_header = nn.Conv1d(REDUCED_EMBED_SIZE, FINAL_CNN_DIM, 1)
        self.conv2_header = nn.Conv1d(REDUCED_EMBED_SIZE, FINAL_CNN_DIM, 2)
        self.conv11_col = nn.Conv2d(REDUCED_EMBED_SIZE, REDUCED_EMBED_SIZE, (1, 1))
        self.conv12_col = nn.Conv2d(REDUCED_EMBED_SIZE, REDUCED_EMBED_SIZE, (1, 2))
        self.conv1_col = nn.Conv1d(2*REDUCED_EMBED_SIZE, FINAL_CNN_DIM, 1)
        self.conv2_col = nn.Conv1d(2*REDUCED_EMBED_SIZE, FINAL_CNN_DIM, 2)
        self.conv11_row = nn.Conv2d(REDUCED_EMBED_SIZE, REDUCED_EMBED_SIZE, (1, 1))
        self.conv12_row = nn.Conv2d(REDUCED_EMBED_SIZE, REDUCED_EMBED_SIZE, (1, 2))
        self.conv1_row = nn.Conv1d(2*REDUCED_EMBED_SIZE, FINAL_CNN_DIM, 1)
        self.conv2_row = nn.Conv1d(2*REDUCED_EMBED_SIZE, FINAL_CNN_DIM, 2)

        self.dense1 = nn.Linear(FINAL_CNN_DIM*(4*NUM_COLS+2*NUM_ROWS-3), FFNN_INTER_DIM)
        self.dense2 = nn.Linear(FFNN_INTER_DIM, NUM_CLASSES)

    def forward(self, x):
        # current shape: (batch_size, num_rows, num_columns, embedding_size)
        # new shape: (batch_size, embedding_size, num_columns, num_rows)
        x = torch.transpose(x, 1, 3)
        
        # dimension reduction
        x = F.relu(self.dimrec_conv1(x))
        x = F.relu(self.dimrec_conv2(x))

        # extract headers, columns, and rows
        headers = x[:, :, :, 0].squeeze(-1)
        cols = x[:, :, :, 1:]
        rows = torch.transpose(cols, 2, 3)

        # process headers
        conv_header_1 = F.relu(self.conv1_header(headers))
        conv_header_2 = F.relu(self.conv2_header(headers))
        header_feats = torch.cat((conv_header_1.flatten(1, -1), conv_header_2.flatten(1, -1)), dim=1)
        
        # process columns
        conv_col_1 = F.relu(self.conv11_col(cols))
        conv_col_1 = F.avg_pool2d(conv_col_1, (1, conv_col_1.size(-1))).squeeze(-1)
        conv_col_2 = F.relu(self.conv12_col(cols))
        conv_col_2 = F.avg_pool2d(conv_col_2, (1, conv_col_2.size(-1))).squeeze(-1)
        cols = torch.cat((conv_col_1, conv_col_2), dim=1)
        conv_col_1 = F.relu(self.conv1_col(cols))
        conv_col_2 = F.relu(self.conv2_col(cols))
        col_feats = torch.cat((conv_col_1.flatten(1, -1), conv_col_2.flatten(1, -1)), dim=1)

        # process rows
        conv_row_1 = F.relu(self.conv11_row(rows))
        conv_row_1 = F.avg_pool2d(conv_row_1, (1, conv_row_1.size(-1))).squeeze(-1)
        conv_row_2 = F.relu(self.conv12_row(rows))
        conv_row_2 = F.avg_pool2d(conv_row_2, (1, conv_row_2.size(-1))).squeeze(-1)
        rows = torch.cat((conv_row_1, conv_row_2), dim=1)
        conv_row_1 = F.relu(self.conv1_row(rows))
        conv_row_2 = F.relu(self.conv2_row(rows))
        row_feats = torch.cat((conv_row_1.flatten(1, -1), conv_row_2.flatten(1, -1)), dim=1)

        # concatenate all features
        # print(header_feats.shape, col_feats.shape, row_feats.shape)
        x = torch.cat((header_feats, col_feats, row_feats), dim=1)
        
        # fully connected layers
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return F.softmax(x, dim=1)
        

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(device)
    table1 = pd.read_csv('Data/explode/train/1/data.csv')
    table2 = pd.read_csv('ATBench/stack/stack_test1/data.csv')
    embedder = Embedding().to(device)
    x = embedder([table1, table2]).to(device).float()
    print(x.shape, x.dtype, x.device)

    model = TableClassifier().to(device)
    y = model(x)
    print(y, y.dtype, y.device)