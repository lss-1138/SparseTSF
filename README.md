# SparseTSF

Welcome to the anonymized repository of the SparseTSF paper: "SparseTSF: Modeling Long-term Time Series Forecasting with *1k* Parameters"

### Model Implementation

The implementation code of SparseTSF is available at:
```
models/SparseTSF.py
```

### Training Scripts

The training scripts (including hyperparameter settings) for replicating the SparseTSF results are available at:

```
scripts/SparsrTSF
```

### Quick Reproduction

You can reproduce all the main results of SparseTSF with the following code snippet.
```
conda create -n SparseTSF python=3.8
conda activate SparseTSF
pip install -r requirements.txt
sh run_all.sh
```

**Thank you again for your efforts and time. We will continue to improve this repository after the paper is accepted.**