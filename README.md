Multi-Manifold Recursive Interaction Learning (MRIL)
==================================================

## 1. Overview

This repository contains the codebase for MRIL, the model introduced in the paper "Orthogonal Multi-Manifold Enriching of Directed Networks" --- The 25th International Conference on Artificial Intelligence and Statistics (AISTATS 2022)



## 2. Setup
### 2.2 Dependencies

```virtualenv -p [PATH to python3.8 binary] mril```

```source mril/bin/activate```

```pip install -r requirements.txt```

### 2.3 Data Preprocessing

The ```data/``` folder contains preprocessing scripts to generate graphs and node embeddings.

Example: Download the PHEME dataset from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078 into ./data/pheme

```python pheme_process.py```

```python embs.py```

```python make_graph.py```


## 3. Usage

### 3.1  ```main.py```

```python main.py```

This script trains models for classification tasks. 
```
 Arguments:
  --h-size DIM          Hidden embedding dimension
  --c C                 Curvature
  --x-size DIM          Input edimension
  --batch-size  BS      Batch size
  --data-dir DIR        Directory for data
  --device DEVICE       Device
  --lr LR               Learning rate
  --dropout DROPOUT     Dropout probability
  --epochs EPOCHS       Maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        L2 regularization strength
  --optimizer OPTIMIZER
                        Which optimizer to use
  --patience PATIENCE   Patience for early stopping
  --beta BETA           CB loss hyperparameter
  --gamma GAMMA         CB loss hyperparameter
  --save                Save computed results
  --save-dir SAVE_DIR   Path to save results
  --min-epochs MIN_EPOCHS
                        Do not early stop before min-epochs
```

## Some of the code was forked from the following repositories
 * [mctorch](https://github.com/mctorch/mctorch)
 * [geoopt](https://github.com/geoopt/geoopt)
 * [hgcn](https://github.com/HazyResearch/hgcn)
 
## References

[1] Sawhney, R., Agarwal, S., Neerkaje, A., and Pathak, K., 2022, March. Orthogonal Multi-Manifold Enriching of Directed Networks. In International Conference on Artificial Intelligence and Statistics. PMLR.
