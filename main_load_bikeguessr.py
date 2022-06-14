import logging
import os
import pickle
import random
# From paper
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import osmnx as ox
import torch
from dgl.data.utils import save_graphs
from dgl.heterograph import DGLHeteroGraph
from networkx.classes.multidigraph import MultiDiGraph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


SELECTED_KEYS = ['oneway', 'lanes', 'highway', 'maxspeed',
                 'length', 'access', 'bridge', 'junction',
                 'width', 'service', 'tunnel']  # not used 'cycleway', 'bycycle']
DEFAULT_VALUES = {'oneway': False, 'lanes': 2, 'highway': 11, 'maxspeed': 50,
                  'length': 0, 'access': 6, 'bridge': 0, 'junction': 0,
                  'width': 2, 'service': 0, 'tunnel': 0}
HIGHWAY_CODING = {'highway': {'primary': 0, 'unclassified': 1, 'tertiary_link': 2, 'secondary': 3,
                              'residential': 4, 'track': 5, 'service': 6, 'trunk': 7, 'tertiary': 8,
                              'primary_link': 9, 'pedestrian': 10, 'path': 11, 'living_street': 12,
                              'trunk_link': 13, 'cycleway': 14, 'bridleway': 15, 'secondary_link': 16},
                  'access': {'customers': 0, 'delivery': 1, 'designated': 2, 'destination': 3,
                             'emergency': 4, 'military': 5, 'no': 6, 'permissive': 7, 'permit': 8, 'yes': 9},
                  'bridge': {'1': 1, 'viaduct': 1, 'yes': 1},
                  'junction': {'yes': 1, 'roundabout': 2, 'y_junction': 3, },
                  'tunnel': {'yes': 1, 'building_passage': 2, 'passage': 3},
                  'service': {'alley': 1, 'bus': 2, 'drive-through': 3, 'driveway': 4,
                              'emergency_access': 5, 'ground': 6, 'parking_aisle': 7, 'spur': 8}}
DATA_INPUT = 'data_raw'
DATA_OUTPUT = 'data_transformed'


def load_directory_bikeguessr(directory: str = None, save: bool = True) -> None:
    logging.info('load bikeguessr directory')
    if directory is None:
        directory = os.path.join(os.getcwd(), DATA_INPUT)
    found_files = list(Path(directory).glob('*.xml'))
    for path in tqdm(found_files):
        load_single_bikeguessr(path, True)
    logging.info('end load bikeguessr directory')


def load_single_bikeguessr(path: str, save: bool = True) -> Tuple[DGLHeteroGraph, StandardScaler]:
    logging.debug('load single bikeguessr')
    bikeguessr_linegraph = _load_transform_linegraph(path)
    bikeguessr_linegraph_with_masks, _, scaler = _create_mask(
        bikeguessr_linegraph)
    if save:
        save_bikeguessr(path, bikeguessr_linegraph_with_masks, scaler)
    logging.debug('end load single bikeguessr')
    return bikeguessr_linegraph_with_masks, scaler


def save_bikeguessr(path: str, graph: DGLHeteroGraph, scaler: StandardScaler) -> None:
    parent = str(Path(path).parent.parent.absolute())
    stem = str(Path(path).stem)
    graph_file = os.path.join(parent, DATA_OUTPUT, stem + '_masks.graph')
    scaler_file = os.path.join(parent, DATA_OUTPUT, stem + '_scaler.pkl')
    save_graphs(graph_file, graph)
    with open(scaler_file, 'wb+') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_transform_linegraph(path: str) -> DGLHeteroGraph:
    raw_graphml = ox.io.load_graphml(path)
    encoded_graphml = _encode_data(raw_graphml)
    seen_values = _get_all_key_and_unique_values(encoded_graphml)
    # print(seen_values['highway'])
    labels_graphml = _generate_cycle_label(
        encoded_graphml, HIGHWAY_CODING['highway'])
    # print(labels_graphml[95584835][6152142174])
    return _convert_nx_to_dgl_as_linegraph(labels_graphml)


def _create_mask(graph: DGLHeteroGraph) -> Tuple[DGLHeteroGraph, List, StandardScaler]:
    num_nodes = graph.num_nodes()

    split_idx = _get_random_split(num_nodes)
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = _preprocess(graph)

    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)

    feat = graph.ndata["feat"]
    feat, scaler = _scale_feats(feat)
    graph.ndata["feat"] = feat

    train_mask = torch.full(
        (num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    num_features = graph.ndata["feat"].shape[1]
    num_classes = 2
    return graph, (num_features, num_classes), scaler


def _encode_data(graph_nx: MultiDiGraph, selected_keys: List = SELECTED_KEYS, default_values: Dict = DEFAULT_VALUES, onehot_key: Dict = HIGHWAY_CODING) -> MultiDiGraph:
    graph_nx_copy = graph_nx.copy()
    for edge in graph_nx.edges():
        for connection in graph_nx[edge[0]][edge[1]].keys():
            graph_edge = graph_nx_copy[edge[0]][edge[1]][connection]
            for key in selected_keys:
                # decide if key exists if not create
                if key in graph_edge.keys():
                    # if value of edge key is a list take first element
                    if type(graph_edge[key]) == list:
                        graph_edge[key] = graph_edge[key][0]

                    if key in onehot_key.keys():
                        if graph_edge[key] in onehot_key[key].keys():
                            graph_edge[key] = onehot_key[key][graph_edge[key]]
                        else:
                            if key in default_values.keys():
                                graph_edge[key] = default_values[key]
                            else:
                                graph_edge[key] = 0

                    if type(graph_edge[key]) == str:
                        try:
                            graph_edge[key] = float(graph_edge[key])
                        except ValueError as e:
                            graph_edge[key] = 0.0

                else:
                    # create key with default values or set to 0
                    if key in default_values.keys():
                        graph_edge[key] = default_values[key]
                    else:
                        graph_edge[key] = 0
    return graph_nx_copy


def _get_all_key_and_unique_values(graph_nx: MultiDiGraph, selected_keys: Dict = SELECTED_KEYS) -> Dict:
    seen_values = {}
    if not selected_keys:
        selected_keys = ['oneway', 'lanes', 'highway', 'maxspeed',
                         'length', 'access', 'bridge', 'junction',
                         'width', 'service', 'tunnel', 'cycleway', 'bycycle']

    # get all values by selected key for each edge
    for edge in graph_nx.edges():
        for connection in graph_nx[edge[0]][edge[1]].keys():
            for key, val in graph_nx[edge[0]][edge[1]][connection].items():
                if key in selected_keys:
                    if key not in seen_values:
                        seen_values[key] = [val]
                    else:
                        if type(val) == list:
                            seen_values[key].extend(val)
                        else:
                            seen_values[key].extend([val])

    for key in seen_values.keys():
        seen_values[key] = set(seen_values[key])
    return seen_values


def _generate_cycle_label(graph_nx: MultiDiGraph, highway_coding: Dict = {}) -> MultiDiGraph:
    graph_nx_copy = graph_nx.copy()
    for edge in graph_nx.edges():
        for connection in graph_nx[edge[0]][edge[1]].keys():
            for key, val in graph_nx[edge[0]][edge[1]][connection].items():
                graph_edge = graph_nx_copy[edge[0]][edge[1]][connection]
                road_type = graph_edge['highway']
                if road_type == 14:
                    graph_edge['label'] = 1
                else:
                    graph_edge['label'] = 0
    return graph_nx_copy


def _convert_nx_to_dgl_as_linegraph(graph_nx: MultiDiGraph, selected_keys: List = SELECTED_KEYS) -> DGLHeteroGraph:
    graph_dgl = dgl.from_networkx(
        graph_nx, edge_attrs=(selected_keys + ['label']))
    graph_dgl_line_graph = dgl.line_graph(graph_dgl)
    # populate linegraph with nodes

    features_to_line_graph = [graph_dgl.edata[key] for key in selected_keys]

    graph_dgl_line_graph.ndata['feat'] = torch.cat(
        features_to_line_graph).reshape((-1, len(selected_keys)))
    graph_dgl_line_graph.ndata['label'] = graph_dgl.edata['label']
    return graph_dgl_line_graph


def _get_random_split(number_of_nodes, train_size_coef=0.05, val_size_coef=0.18, test_size_coef=0.37):
    split_idx = {}
    train_size = int(number_of_nodes * train_size_coef)
    val_size = int(number_of_nodes * val_size_coef)
    test_size = int(number_of_nodes * test_size_coef)
    split_idx['train'] = random.sample(range(0, number_of_nodes), train_size)
    split_idx['train'].sort()
    split_idx['valid'] = random.sample(range(0, number_of_nodes), val_size)
    split_idx['valid'].sort()
    split_idx['test'] = random.sample(range(0, number_of_nodes), test_size)
    split_idx['test'].sort()

    return split_idx


def _scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats, scaler


def _preprocess(graph):
    feat = graph.ndata["feat"]
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


if __name__ == "__main__":
    load_directory_bikeguessr()
