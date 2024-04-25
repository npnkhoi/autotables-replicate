"""
Train the model
"""

from src.model import Embedding, TableClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data import TableDataset, test_data
from torch.utils.data import random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
dataset = TableDataset(test_data)
batch_size = 4
epochs = 3

def train():
    # Split the dataset
    # Determine the lengths of the splits
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    print(f'Train size: {train_size}, Test size: {test_size}')

    # Split the dataset
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Initialize the model
    embedding = Embedding().to(device)
    model = TableClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    model.train()
    for epoch in range(epochs):
        for i, (Xs, ys) in enumerate(train_loader):
            dfs = [pd.read_csv(path) for path in Xs]
            Xs = embedding(dfs).float().to(device)
            ys = ys.clone().detach().to(device)

            optimizer.zero_grad()
            output = model(Xs)
            loss = F.cross_entropy(output, ys)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}')
    
        # Test the model
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for Xs, ys in test_loader:
                dfs = [pd.read_csv(path) for path in Xs]
                Xs = embedding(dfs).float().to(device)
                output = model(Xs)
                y_true.extend(ys)
                y_pred.extend(output.argmax(dim=1).tolist())

        print(f'Accuracy: {accuracy_score(y_true, y_pred)}')


# def force_cudnn_initialization():
#     # https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

if __name__ == "__main__":
    print(device)
    # force_cudnn_initialization()
    train()