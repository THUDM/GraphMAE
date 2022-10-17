<h1> GraphMAE: Self-Supervised Masked Graph Autoencoders</h1>

**Transfer learning for moleculer property prediction**

Datasets for molecular property prediction can be found [here](https://github.com/snap-stanford/pretrain-gnns#dataset-download) (This [link](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) for downloading).

<h2> Dependencies </h2>

pytorch >= 1.8.0

torch_geometric >= 2.0.3

rdkit >= 2019.03.1.0

tqdm >= 4.31

<h2> Pre-training and fine-tuning </h2>

**1. pre-training**

```bash
python pretraining.py
```

**2. fine-tuning**

```bash
python finetune.py --input_model_file <model_path> --dataset <dataset_name>
```

Results in the paper can be reproduced by running `sh finetune.sh <dataset_name>` using the pre-trained model in `./init_weights/pretrained.pth`. Most hyper-parameters are shared across datasets. The differences can be found in `finetuning.py`.



<h2>Acknowledgements</h2>

The implementation is based on the codes in Hu et al: [Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns)