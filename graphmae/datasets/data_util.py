from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops

from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_name):
    #transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = Planetoid("", dataset_name, transform=T.NormalizeFeatures())
    graph = dataset[0]
    #graph.edge_index = remove_self_loops(graph.edge_index)[0]
    #graph.edge_index = add_self_loops(graph.edge_index)[0]

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)

