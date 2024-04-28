# autotables-replicate

Replication code of the paper [Auto-Tables: Synthesizing Multi-Step Transformations to Relationalize Tables without Using Examples](https://arxiv.org/abs/2307.14565).

## Install

See `requirements.txt`

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
    logs/result/model.pth
    --device cuda
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
