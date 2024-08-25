## SkipPool: Improved Sparse Hierarchical Graph Pooling with Differentiable Exploration

### Folder Descriptions

* `classification` - this folder contains code required to reproduce experiments in the section 4.1
  * `Full` - this subfolder contains code to reproduce experiments on SkipPool-Full
  * `Fixed` - this subfolder contains code to reproduce experiments on standard SkipPool
* `autoencoder` - this folder contains code to reproduce the experiment in the section 4.2 

## Training models from scratch
### Standard SkipPool

* Go to `classification/Fixed/model`.
* For example run the following to train on DD:
```
python main.py --dataset --nhid 128 --batch_size 64 --lr 1e-3 --weight_decay 1e-4 --dropout_ratio 0.4
```
* Data required for this experiment should be available in `classification/data` directory.
* The trained models are inside `[dataset_name]/rep_0/fold_[fold no.]`, accuracy for each fold is logged in file `folds_res_log.txt`.
* Average accuracy for the rep 0 is logged in `rep_res_log.txt`.


### SkipPool-Full 

* Go to `classification/Full/model`.
* For example run the following to train on DD:
```
python main.py --dataset DD --fixStride 5 --batch_size 128 --nhid 64 --lr 1e-4 --conv gcn
```
* Data required for this experiment should be available in `classification/data` dir.
* The trained models are inside `[dataset_name]/fold_[fold no.]`, accuracy for each fold is logged in file `folds_res_log.txt`

#### For both SkipPool & SkipPool-Full you can specify the number process for each pooling layer by a list. use `--p-list` command. 

* For example run the following to have 2,2 number of processes in respective pooling layers:
```
python main.py --dataset DD --fixStride 5 --batch_size 128 --nhid 64 --lr 1e-4 --conv gcn --p-list 2,2
```

#### Training GAE
* Go to  `autoencoder` directory.
* If you want to train with SkipPool-Full then use `--full` command. Otherwise standard SkipPool is used.

Example:
```
python run_ae.py --dataset [graph name] --full --fixStride 5
python run_ae.py --dataset Ring --full --fixStride 5
```
* MODELNET40 from which Airplane, Car, Guitar, Person are obtained, is not readily available. If one of these Graphs are needed they it will be downloaded.
* Data used to train GAE are at `autoencoder/data`
* Training models will be at `[method]/[graph_name]/rep_0`, method is either Full or Fixed. 

### Dependencies

- Pytorch (2.0.1)
- Pytorch_Scatter (2.1.1)
- torch_sparse (0.6.17)
- Pytorch_Geometric (2.3.1)
- PyGSP (0.5.1)


### Hyperparameters to reproduce reported scores in the paper

#### SkipPool

| Dataset | Hidden Dimension | Batch Size | Learning Rate | Weight Decay | Dropout 
|---|---|---|---|---|---|
| DD | 128 | 64 | 1e-3 | 1e-4 | 0.4
| PROTEINS | 64 | 64 | 1e-3 | 1e-4 | 0.5
| NCI1 | 128 | 32 | 5e-4 | 1e-4 | 0.0
| NCI109 | 128 | 32 | 5e-4 | 1e-4 | 0.2
| FRANKENSTEIN | 32 | 32 | 1e-3 | 1e-4 | 0.0

#### SkipPool-Full

| Dataset | Hidden Dimension | Batch Size | Learning Rate |
|---|---|---|---|
| DD | 64 | 128 | 1e-4 |
| REDDIT-B | 128 | 32 | 1e-3 |
