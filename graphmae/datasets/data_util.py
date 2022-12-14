from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree

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
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root="./data")
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


def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(root="./data", name=dataset_name)
    dataset = list(dataset)
    graph = dataset[0]


    if graph.x == None:
        if graph.y and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g in dataset:
                feature_dim = max(feature_dim, int(g.y.max().item()))
            
            feature_dim += 1
            for i, g in enumerate(dataset):
                node_label = g.y.view(-1)
                feat = F.one_hot(node_label, num_classes=int(feature_dim)).float()
                dataset[i].x = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g in dataset:
                feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
                degrees.extend(degree(g.edge_index[0]).tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for i, g in enumerate(dataset):
                degrees = degree(g.edge_index[0])
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                g.x = feat
                dataset[i] = g

    else:
        print("******** Use `attr` as node features ********")
    feature_dim = int(graph.num_features)

    labels = torch.tensor([x.y for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    for i, g in enumerate(dataset):
        dataset[i].edge_index = remove_self_loops(dataset[i].edge_index)[0]
        dataset[i].edge_index = add_self_loops(dataset[i].edge_index)[0]
    #dataset = [(g, g.y) for g in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    return dataset, (feature_dim, num_classes)
