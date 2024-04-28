"""
Train the model

Usage: python -m src.train \
    --outdir logs/test2 \
    --batch_size 8 \
    --epochs 10 \
    --device cuda \
    --embedding_model_path logs/test2 \
    --lr 4e-3

Author: Khoi Nguyen
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
from src.data import get_test_data, get_train_val_data
from argparse import ArgumentParser
import json
import os
# from sklearn.model_selection import train_test_split
# from torch.utils.data import random_split

def train(args):
    # Get args
    outdir = args.outdir
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    embedding_model_path = args.embedding_model_path

    # Set device
    device = torch.device(device)

    # Create outdir
    os.makedirs(outdir, exist_ok=True)

    # Load data
    train_set, val_set = get_train_val_data()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print(f'Train set size: {len(train_set)}, Val set size: {len(val_set)}')

    # Initialize the model
    embedding = Embedding(embedding_model_path).to(device)
    model = TableClassifier().to(device)
    if args.classifier_model_path is not None:
        model.load_state_dict(torch.load(args.classifier_model_path))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    logs = {
        'args': args.__dict__,
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    best_val_accuracy = 0

    for epoch in range(epochs):
        # Train the model
        train_loss = 0
        for i, (Xs, ys) in enumerate(train_loader):
            model.train()
            dfs = [pd.read_csv(path) for path in Xs]
            Xs = embedding(dfs).float().to(device)
            ys = ys.clone().detach().to(device)

            optimizer.zero_grad()
            output = model(Xs)
            loss = F.cross_entropy(output, ys)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f'Epoch {epoch}, Iter {i}, Loss: {loss.item()}')
        
        train_loss /= len(train_loader)

        # Evaluate the model
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        for i, (Xs, ys) in enumerate(val_loader):
            dfs = [pd.read_csv(path) for path in Xs]
            Xs = embedding(dfs).float().to(device)
            ys = ys.clone().detach().to(device)

            output = model(Xs)
            loss = F.cross_entropy(output, ys)
            val_loss += loss.item()

            val_preds += output.argmax(dim=1).tolist()
            val_true += ys.tolist()
        
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_true, val_preds)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')
        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['val_accuracy'].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            torch.save(model.state_dict(), f'{outdir}/model.pth')
            best_val_accuracy = val_accuracy
    
    # save logs
    json.dump(logs, open(f'{outdir}/logs.json', 'w'), indent=2)
    print(f'Best val accuracy: {best_val_accuracy}')


# def force_cudnn_initialization():
#     # https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

if __name__ == "__main__":
    # parse args
    parser = ArgumentParser()
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embedding_model_path', type=str, default='sentence-transformers/paraphrase-MiniLM-L3-v2')
    parser.add_argument('--classifier_model_path', type=str, default=None, help='path to the classifier model\'s state dict FILE. None means training from scratch.')
    args = parser.parse_args()
    
    print(args)
    train(args)