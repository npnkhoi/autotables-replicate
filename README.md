# Replicating AutoTable s

Replication code of the paper - [Auto-Tables: Synthesizing Multi-Step Transformations to Relationalize Tables without Using Examples](https://arxiv.org/abs/2307.14565).

## Overview

AutoTables is an algorithm to automatically transform tables from non-relational form to the relational one.

In this repo, we only replicate the transformation predictor, which is a 7-class classifier: `explode`, `ffill`, `pivot`, `stack`, `subtitle`, `transpose`, and `wide_to_long`. For simplicity, we excluded `multistep` operator and do not predict the arguments for the transformation functions.

While the original model was trained on more than 100K tables, the training data was not released. Therefore, we synthesized the data ourselves using "inverse operators", resulting in over 1K tables.

Full technical report that includes evaluation results will be updated later.

## Install

```bash
pip install -r requirements.txt
```

Important notice: Our training code requires GPU. Consequently, the PyTorch version needs to match with the CUDA version on your machine. Our specified Pytorch was compiled with CUDA 11.8.

## Train

Run
```bash
python -m src.train \
    --outdir logs/result \
    --batch_size 8 \
    --epochs 10 \
    --device cuda
```

## Evaluation
After training, run
```bash
python -m src.eval \
    logs/result/model.pth \
    --device cuda
```

However, if you just want to run our final model on CPU, run:
```
python -m src.eval \
    logs/final/model.pth \
    --batch_size 4 \
    --device cpu
```

## Datasets
We generated datasets from doing inverse operations on the relational tables to make it into non relational tables. We used the tables from AutoTable Benchmark dataset and augmented the tables to increase the size of dataset to train.  
Operations we used:
- stack
- wide_to_long
- transpose
- pivot
- explode
- ffill
- subtitle

##### Directory Structure:
```
.
+-- Data
|   +-- operation1
|       +-- train
|           +-- Folder(x)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           +-- Folder(x+1)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           .
|           .
|           .
|       +-- test
|           +-- Folder(x)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           +-- Folder(x+1)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           .
|           .
|           .
|   +-- operation2
|       +-- train
|           +-- Folder(x)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           +-- Folder(x+1)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           .
|           .
|           .
|       +-- test
|           +-- Folder(x)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           +-- Folder(x+1)
|               +-- data.csv (input)
|               +-- gt.csv (output)
|           .
|           .
|           .
```
