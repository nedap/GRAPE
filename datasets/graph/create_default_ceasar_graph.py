from tqdm import tqdm
from pathlib import Path

import pytorch_lightning as pl
from utils.misc import *
from datasets.pytorch_lightning import CeasarDataModule
from models.pytorch_lightning import MAELightningModule, GAELightningModule

from utils import parser
from utils.config import *


def get_default_features(args, cfg, nodes):
    """
    Calculate node features based on encoder output and update super node features by averaging constituent node features.

    :param args: Execution arguments.
    :param cfg: Configuration settings.
    :param nodes: List of nodes in the graph.
    """

    # Define super node relationships and compute their features
    super_nodes = {
        20: [1, 9, 2],  # torso is made up of [lattisimus, pectoralis, abdominus]
        19: [7, 3, 8],  # arms is made up of [deltoid, bicep, triceps]
        18: [10],  # hip is made up of [gluteus]
        17: [11, 6],  # lower_leg is made up of [calve, patella]
        16: [5, 4],  # thigh is made up of [hamstrings, quadriceps]
        15: [19, 20],  # upper_body is made up of [arms, torso]
        14: [16, 17, 18],  # lower_body is made up of [thigh, lower_leg, hip]
        13: [14, 15]  # body is made up of [lower_body, upper_body]
    }

    features = {node['label']: [] for node in nodes}
    average_features = {}

    # Load and freeze the encoder
    pretrained_frozen_encoder = MAELightningModule.load_and_freeze_encoder(cfg.group_and_encode_model.pretrained_ckpnt,
                                                                           cfg, args)
    group_and_encode = GAELightningModule(cfg, args=args, pretrained_encoder=pretrained_frozen_encoder, base_type=False)

    # Load the data module
    data_module = CeasarDataModule(cfg=cfg, args=args)

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

    # Loop through all predictions to get all features
    for encoded_batch, labels_batch, _ in tqdm(predictions):
        for encoded_pcd, labels in zip(encoded_batch, labels_batch):
            for encoded_patch, label in zip(encoded_pcd, labels):
                features[label.item()].append(encoded_patch)

    # Compute the mean feature for each leaf node, i.e. 1-12
    for node_label, node_features in features.items():
        if node_features and node_label <= 12:
            mean_feature = np.mean(node_features, axis=0)
            average_features[node_label] = mean_feature.tolist()

    # calculate the mean feature for each super node
    for super_node, constituents in super_nodes.items():
        constituent_features = [average_features[node] for node in constituents]
        average_features[super_node] = np.mean(constituent_features, axis=0).tolist()

    for node in features:
        # if np.shape(features[node]) == (0,):
        features[node] = average_features[node]

    # Update node features in the graph
    for node in nodes:
        # if node['feature'] is None:
        node['feature'] = features[node['label']]

    print("final features:")
    for node in nodes:
        print(f"Node: {node['name']} - {node['parent_node']}: {np.shape(node['feature'])}")

    # make 3 copies of tabletop_connector, 

    return nodes


def build_default_ceasar_graph(args, cfg):
    nodes = [
        {'name': 'lattisimus', 'label': 1, 'feature': None, 'parent_node': 20},
        {'name': 'abdominus', 'label': 2, 'feature': None, 'parent_node': 20},
        {'name': 'bicep', 'label': 3, 'feature': None, 'parent_node': 19},
        {'name': 'quadriceps', 'label': 4, 'feature': None, 'parent_node': 16},
        {'name': 'hamstrings', 'label': 5, 'feature': None, 'parent_node': 16},
        {'name': 'patella', 'label': 6, 'feature': None, 'parent_node': 17},
        {'name': 'deltoid', 'label': 7, 'feature': None, 'parent_node': 19},
        {'name': 'triceps', 'label': 8, 'feature': None, 'parent_node': 19},
        {'name': 'pectoralis', 'label': 9, 'feature': None, 'parent_node': 20},
        {'name': 'gluteus', 'label': 10, 'feature': None, 'parent_node': 18},
        {'name': 'calve', 'label': 11, 'feature': None, 'parent_node': 17},
        {'name': 'undefined', 'label': 12, 'feature': None, 'parent_node': None},

        # these are the supernodes, i.e. no part
        {'name': 'body', 'label': 13, 'feature': None, 'parent_node': None, },
        {'name': 'lower_body', 'label': 14, 'feature': None, 'parent_node': 13, },
        {'name': 'upper_body', 'label': 15, 'feature': None, 'parent_node': 13, },
        {'name': 'thigh', 'label': 16, 'feature': None, 'parent_node': 14, },
        {'name': 'lower_leg', 'label': 17, 'feature': None, 'parent_node': 14, },
        {'name': 'hip', 'label': 18, 'feature': None, 'parent_node': 14, },
        {'name': 'arms', 'label': 19, 'feature': None, 'parent_node': 15, },
        {'name': 'torso', 'label': 20, 'feature': None, 'parent_node': 15, },

    ]

    graph = {
        'nodes': nodes,
        'hierarchy_edges': []
    }

    graph['nodes'] = get_default_features(args, cfg, nodes)

    current_dir = Path.cwd()

    with open(current_dir / 'datasets/graph/default_ceasar_graph.yaml', 'w') as file:
        yaml.dump(graph, file, sort_keys=False)
