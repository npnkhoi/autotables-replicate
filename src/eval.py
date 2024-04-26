"""
Get a saved model, get accuracy on test set

python -m src.eval --outdir ...
"""

from src.model import Embedding, TableClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from src.data import get_test_data
from argparse import ArgumentParser
import json
import os

def eval(args):
    # Get args
    batch_size = args.batch_size
    device = args.device

    # Set device
    device = torch.device(device)

    # Load data
    test_set = get_test_data()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print(f'Test set size: {len(test_set)}')

    # Load the model
    embedding = Embedding().to(device)
    model = TableClassifier().to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Evaluate the model
    y_true = []
    y_pred = []
    for Xs, ys in test_loader:
        dfs = [pd.read_csv(path) for path in Xs]
        Xs = embedding(dfs).float().to(device)
        ys = ys.to(device)
        with torch.no_grad():
            logits = model(Xs)
            y_true.extend(ys.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test accuracy: {accuracy}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    eval(args)