from tqdm import tqdm
from pathlib import Path

import pytorch_lightning as pl
from utils.misc import *
from datasets.pytorch_lightning import PartNetDataModule
from models.pytorch_lightning import MAELightningModule, GAELightningModule

from utils import parser
from utils.config import *


# this is the default table graph based on 'notebooks/img/default_graph.png'
# NOTE there are three node types: object, sub_structure, part.


def get_default_features(args, cfg, nodes):
    """
    Calculate node features based on encoder output and update super node features by averaging constituent node features.

    :param args: Execution arguments.
    :param cfg: Configuration settings.
    :param nodes: List of nodes in the graph.
    """

    base_types = {
        "regular_leg_base": 28,
        "drawer_base": 26,
        "pedestal_base": 25,
        "star_leg_base": 29
    }

    # Define super node relationships and compute their features
    super_nodes = {
        29: [1, 15],  # star_leg_base is made up of [leg, central_support]
        28: [7, 17, 1, 11, 8],  # regular_leg_base is made up of [foot, runner, leg, bar_stretcher, tabletop_connector]
        27: [14, 13, 2],  # drawer is made up of [drawer_side, drawer_front, drawer_bottom]
        26: [10, 7, 18, 5, 20, 3, 1, 11, 16, 8, 6],
        # drawer_base is made up of [vertical_front_panel, foot, bottom_panel, vertical_divider_panel, shelf, vertical_side_panel, leg, bar_stretcher, back_panel, tabletop_connector, cabinet_door_surface]
        25: [8, 9],  # pedestal_base is made up of [tabletop_connector, pedestal]
        24: [25, 26, 28, 29],  # table_base is made up of [pedestal_base, drawer_base, regular_leg_base, star_leg_base]
        23: [4, 12, 0, 19],  # table_top is made up of [circle, board, glass, bar]
        22: [23, 24]  # table is made up of [table_top, table_base]
    }

    features_per_base = {base_type: {node['label']: [] for node in nodes} for base_type in list(base_types.values())}
    features_global = {node['label']: [] for node in nodes}
    average_features = {}

    # Load and freeze the encoder
    pretrained_frozen_encoder = MAELightningModule.load_and_freeze_encoder(cfg.group_and_encode_model.pretrained_ckpnt,
                                                                           cfg, args)
    group_and_encode = GAELightningModule(cfg, args=args, pretrained_encoder=pretrained_frozen_encoder, base_type=True)

    # Load the data module
    data_module = PartNetDataModule(cfg=cfg, args=args)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[int(cfg.device.device_id)],
        max_epochs=1,  # one epoch to extract all latents
        logger=None,
        default_root_dir=args.experiment_path,
    )

    # Inference
    predictions = trainer.predict(model=group_and_encode, datamodule=data_module)

    # Construct the graph dataset:
    graph_data_path = Path(cfg.graph_data_path)

    # Create the graph data path if it does not yet exist.
    graph_data_path.mkdir(parents=True, exist_ok=True)

    # Loop through all predictions to calculate features
    for encoded_batch, labels_batch, _, _, _, base_types_batch in tqdm(predictions):
        for encoded_pcd, labels, base_type in zip(encoded_batch, labels_batch, base_types_batch):
            for encoded_patch, label in zip(encoded_pcd, labels):
                features_per_base[base_types[base_type]][label.item()].append(encoded_patch)
                features_global[label.item()].append(encoded_patch)

    # Compute the mean feature for each leaf node
    for base_type in list(base_types.values()):
        for node_label, node_features in features_per_base[base_type].items():
            if node_features:
                mean_feature = np.mean(node_features, axis=0)
                features_per_base[base_type][node_label] = mean_feature.tolist()
                features_global[node_label] = mean_feature.tolist()
            else:
                features_per_base[base_type][node_label] = None  # Handle empty feature lists explicitly

    # calculate the mean feature for each super node
    for super_node, constituents in super_nodes.items():
        # if star_leg_base, regular_leg_base, drawer, drawer_base, pedestal_base (direct parents of the leaf nodes)
        if super_node in [25, 26, 27, 28, 29]:
            # If drawer, then we need to use the drawer_base features
            if super_node == 27:
                constituent_features = [features_per_base[26][node] for node in constituents]
            else:
                constituent_features = [features_per_base[super_node][node] for node in constituents]
            average_features[super_node] = np.mean(constituent_features, axis=0).tolist()
        # if table_top we collect features from all base types
        if super_node in [23]:
            constituent_features = []
            for base_type, base_node in base_types.items():
                for node in constituents:
                    constituent_features.append(features_per_base[base_node][node])
            average_features[super_node] = np.mean(constituent_features, axis=0).tolist()
        # if table or table_base
        elif super_node in [22, 24]:
            # These nodes are parents of parent nodes.
            constituent_features = [average_features[node] for node in constituents]
            average_features[super_node] = np.mean(constituent_features, axis=0).tolist()
        else:
            ValueError(f"Super node {super_node} should not exist")

    for feature in average_features:
        print(f"Feature: {feature}: {np.shape(average_features[feature])}")

    for node in features_global:
        if np.shape(features_global[node]) == (0,):
            features_global[node] = average_features[node]

    # print("final features:")
    # for feature in features_global:
    #     print(f"Feature: {feature}: {np.shape(features_global[feature])}")

    # Update node features in the graph
    for node in nodes:
        # if node['feature'] is None:
        node['feature'] = features_global[node['label']]

    print("final features:")
    for node in nodes:
        print(f"Node: {node['name']} - {node['parent_node']}: {np.shape(node['feature'])}")

    # make 3 copies of tabletop_connector, 

    return nodes


# NOTE if you want to create a graph with double connection for the shared leaf nodes, simply remove the duplicated
# here, and use the old create_edge_index
def build_default_table_graph(args, cfg):
    nodes = [
        # these are the actual part nodes, labels conform to /srv/healthcare/datascience/data/part-net/data_table/all_unique_integer_label_pairs.csv
        {'name': 'glass', 'label': 0, 'feature': None, 'parent_node': 23, 'pos_embedding': None},
        # 3 copies of leg
        {'name': 'leg', 'label': 1, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'leg', 'label': 1, 'feature': None, 'parent_node': 28, 'pos_embedding': None},
        {'name': 'leg', 'label': 1, 'feature': None, 'parent_node': 29, 'pos_embedding': None},

        {'name': 'drawer_bottom', 'label': 2, 'feature': None, 'parent_node': 27, 'pos_embedding': None},
        {'name': 'vertical_side_panel', 'label': 3, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'circle', 'label': 4, 'feature': None, 'parent_node': 23, 'pos_embedding': None},
        {'name': 'vertical_divider_panel', 'label': 5, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'cabinet_door_surface', 'label': 6, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        # 2 copies of foot
        {'name': 'foot', 'label': 7, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'foot', 'label': 7, 'feature': None, 'parent_node': 28, 'pos_embedding': None},

        # 3 copies of tabletop_connector
        {'name': 'tabletop_connector', 'label': 8, 'feature': None, 'parent_node': 25, 'pos_embedding': None},
        {'name': 'tabletop_connector', 'label': 8, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'tabletop_connector', 'label': 8, 'feature': None, 'parent_node': 28, 'pos_embedding': None},

        {'name': 'pedestal', 'label': 9, 'feature': None, 'parent_node': 25, 'pos_embedding': None},
        {'name': 'vertical_front_panel', 'label': 10, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        # 2 copies of bar_stretcher
        {'name': 'bar_stretcher', 'label': 11, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'bar_stretcher', 'label': 11, 'feature': None, 'parent_node': 28, 'pos_embedding': None},

        {'name': 'board', 'label': 12, 'feature': None, 'parent_node': 23, 'pos_embedding': None},
        {'name': 'drawer_front', 'label': 13, 'feature': None, 'parent_node': 27, 'pos_embedding': None},
        {'name': 'drawer_side', 'label': 14, 'feature': None, 'parent_node': 27, 'pos_embedding': None},
        {'name': 'central_support', 'label': 15, 'feature': None, 'parent_node': 29, 'pos_embedding': None},
        {'name': 'back_panel', 'label': 16, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'runner', 'label': 17, 'feature': None, 'parent_node': 28, 'pos_embedding': None},
        {'name': 'bottom_panel', 'label': 18, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'bar', 'label': 19, 'feature': None, 'parent_node': 23, 'pos_embedding': None},
        {'name': 'shelf', 'label': 20, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'miscellaneous', 'label': 21, 'feature': None, 'parent_node': None, 'pos_embedding': None},

        # these are the supernodes, i.e. no part
        {'name': 'regular_table', 'label': 22, 'feature': None, 'parent_node': None, 'pos_embedding': None},
        {'name': 'table_top', 'label': 23, 'feature': None, 'parent_node': 22, 'pos_embedding': None},
        {'name': 'table_base', 'label': 24, 'feature': None, 'parent_node': 22, 'pos_embedding': None},
        {'name': 'pedestal_base', 'label': 25, 'feature': None, 'parent_node': 24, 'pos_embedding': None},
        {'name': 'drawer_base', 'label': 26, 'feature': None, 'parent_node': 24, 'pos_embedding': None},
        {'name': 'drawer', 'label': 27, 'feature': None, 'parent_node': 26, 'pos_embedding': None},
        {'name': 'regular_leg_base', 'label': 28, 'feature': None, 'parent_node': 24, 'pos_embedding': None},
        {'name': 'star_leg_base', 'label': 29, 'feature': None, 'parent_node': 24, 'pos_embedding': None},
    ]

    graph = {
        'nodes': nodes,
        'hierarchy_edges': []
    }

    graph['nodes'] = get_default_features(args, cfg, nodes)

    current_dir = Path.cwd()

    with open(current_dir / 'datasets/graph/default_table_graph.yaml', 'w') as file:
        yaml.dump(graph, file, sort_keys=False)
