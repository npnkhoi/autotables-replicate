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

