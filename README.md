<h1> GraphMAE: Self-Supervised Masked Graph Autoencoders </h1>

Implementation for KDD22 paper:  [*GraphMAE: Self-Supervised Masked Graph Autoencoders*](https://arxiv.org/abs/2205.10803).

<h2>Dependencies </h2>

* Python >= 3.7
* [Pytorch](https://pytorch.org/) >= 1.9.0 
* [dgl](https://www.dgl.ai/) >= 0.7.2
* pyyaml == 5.4.1

<h2>Quick Start </h2>

For quick start, you could run the scripts: 

**Node classification**

```bash
sh scripts/run_transductive.sh <dataset_name> <gpu_id> # for transductive node classification
# example: sh scripts/run_transductive.sh cora/citeseer/pubmed/ogbn-arxiv 0
sh scripts/run_inductive.sh <dataset_name> <gpu_id> # for inductive node classification
# example: sh scripts/run_inductive.sh reddit/ppi 0

# Or you could run the code manually:
# for transductive node classification
python main_transductive.py --dataset cora --encoder gat --decoder gat --seed 0 --device 0
# for inductive node classification
python main_inductive.py --dataset ppi --encoder gat --decoder gat --seed 0 --device 0
```

Supported datasets:

* transductive node classification:  `cora`, `citeseer`, `pubmed`, `ogbn-arxiv`
* inductive node classification: `ppi`, `reddit` 

Run the scripts provided or add `--use_cfg` in command to reproduce the reported results.



**Graph classification**

```bash
sh scripts/run_graph.sh <dataset_name> <gpu_id>
# example: sh scripts/run_graph.sh mutag/imdb-b/imdb-m/proteins/... 0 

# Or you could run the code manually:
python main_graph.py --dataset IMDB-BINARY --encoder gin --decoder gin --seed 0 --device 0
```

Supported datasets: 

- `IMDB-BINARY`, `IMDB-MULTI`, `PROTEINS`, `MUTAG`, `NCI1`, `REDDIT-BINERY`, `COLLAB`

Run the scripts provided or add `--use_cfg` in command to reproduce the reported results.



**Molecular Property Prediction**

Please refer to codes in `./chem` for *molecular property prediction*.

<h2> Datasets </h2>

Datasets used in node classification and graph classification will be downloaded automatically from https://www.dgl.ai/ when running the code. 
