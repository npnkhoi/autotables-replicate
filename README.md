# autotables-replicate

### installations:
- pandas
- torch (with appropriate version for your CUDA)

### datasets:
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
