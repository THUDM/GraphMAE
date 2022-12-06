from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected

from ogb.nodeproppred import PygNodePropPredDataset

from sklearn.preprocessing import StandardScaler

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def load_dataset(dataset_name):
    if dataset_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root="/mnt/vepfs/yufei/dataset/")
        graph = dataset[0]
        num_nodes = graph.x.shape[0]
        graph.edge_index = to_undirected(graph.edge_index)
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)
        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.train_mask, graph.val_mask, graph.test_mask = train_mask, val_mask, test_mask
        graph.y = graph.y.view(-1)
        graph.x = scale_feats(graph.x)
    else:
        dataset = Planetoid("", dataset_name, transform=T.NormalizeFeatures())
        graph = dataset[0]
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)

