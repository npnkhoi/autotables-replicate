"""
Given the outdir, plot the training logs
"""

import json
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def plot_training(args):
    # Get args
    outdir = args.outdir

    # Load the logs
    with open(os.path.join(outdir, 'logs.json'), 'r') as f:
        logs = json.load(f)

    # Plot the logs
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(logs['train_loss'], label='train')
    axes[0].plot(logs['val_loss'], label='val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(logs['train_accuracy'], label='train')
    axes[1].plot(logs['val_accuracy'], label='val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training.png'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    plot_training(args)