import numpy as np
import copy
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from utils.misc import *
from datasets.pytorch_lightning import PartNetDataModule, CeasarDataModule
from models.pytorch_lightning import MAELightningModule, GAELightningModule


def build_gnn_ds_partnet(args, cfg, wandb_logger, terminal_logger):
    """
    Loop through the entire dataset, extract latent representation of each sample from PointMAE, 
    then save as a new dataset.

    :param cfg: The configuration.
    """
    # Load and freeze the encoder
    pretrained_frozen_encoder =  MAELightningModule.load_and_freeze_encoder(cfg.group_and_encode_model.pretrained_ckpnt, cfg, args)
    group_and_encode = GAELightningModule(cfg, args=args, pretrained_encoder=pretrained_frozen_encoder, base_type=True)
    # load the data module
    data_module = PartNetDataModule(cfg=cfg, args=args)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[int(cfg.device.device_id)], 
        max_epochs=1, # one epoch to extract all latents
        logger=wandb_logger, 
        default_root_dir=args.experiment_path,
    )
    
    # fit the model
    predictions = trainer.predict(model=group_and_encode, datamodule=data_module)
    
    # construct the graph dataset:
    graph_data_path = Path(cfg.graph_data_path)
    default_graph = load_default_graph(cfg.default_graph)

    # create the graph data path if it does not yet exist. 
    graph_data_path.mkdir(parents=True, exist_ok=True)

    # loop through all predictions
    print("creating graph dataset...")
    sample = 0
    for encoded_batch, labels_batch, base_type_batch in tqdm(predictions):
        for encoded_pcd, labels, base_type in zip(encoded_batch, labels_batch, base_type_batch):
            file_name = str(sample)
            sample += 1
            nodes = copy.deepcopy(default_graph['nodes'])
            # populate the nodes
            for encoded_patch, label in zip(encoded_pcd, labels):
                encoded_patch = encoded_patch.detach().cpu().numpy()
                label = label.detach().cpu().numpy()[0]
                nodes = add_node_feature_partnet(nodes=nodes, label=label, feature=encoded_patch, base_type=base_type)

            nodes = nodes_to_numpy(nodes=nodes, feature_dim=cfg.model.transformer_config.trans_dim, cfg=cfg)
            # prevent information/label leakage in the ordering of nodes/features
            if cfg.shuffle_nodes:  
                np.random.shuffle(nodes)

            # allow max 1 edge per leaf node and parent
            hierarchy_edges = generate_edges(nodes=nodes)
                
            # save the graph:
            save_graph(nodes=nodes, hierarchy_edges=hierarchy_edges, path=graph_data_path / file_name)
    print(f'{len([p for p in graph_data_path.iterdir() if p.is_dir()])} graphs created')


def build_gnn_ds_ceasar(args, cfg, wandb_logger, terminal_logger):
    """
    Loop through the entire dataset, extract latent representation of each sample from PointMAE, 
    then save as a new dataset.

    :param cfg: The configuration.
    """
    # Load and freeze the encoder
    pretrained_frozen_encoder =  MAELightningModule.load_and_freeze_encoder(cfg.group_and_encode_model.pretrained_ckpnt, cfg, args)
    group_and_encode = GAELightningModule(cfg, args=args, pretrained_encoder=pretrained_frozen_encoder, base_type=False, mask=True)
    # load the data module
    data_module = CeasarDataModule(cfg=cfg, args=args)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[int(cfg.device.device_id)], 
        max_epochs=1, # one epoch to extract all latents
        logger=wandb_logger, 
        default_root_dir=args.experiment_path,
    )
    
    # fit the model
    predictions = trainer.predict(model=group_and_encode, datamodule=data_module)
    
    # construct the graph dataset:
    graph_data_path = Path(cfg.graph_data_path)
    default_graph = load_default_graph(cfg.default_graph)

    # create the graph data path if it does not yet exist. 
    graph_data_path.mkdir(parents=True, exist_ok=True)

    # loop through all predictions
    print("creating graph dataset...")
    sample = 0
    for encoded_batch, labels_batch in tqdm(predictions):
        for encoded_pcd, labels in zip(encoded_batch, labels_batch):
            file_name = str(sample)
            sample += 1
            nodes = copy.deepcopy(default_graph['nodes']) 
            # populate the nodes
            for encoded_patch, label in zip(encoded_pcd, labels):
                encoded_patch = encoded_patch.detach().cpu().numpy()
                label = label.detach().cpu().numpy()[0]
                nodes = add_node_feature_ceasar(nodes=nodes, label=label, feature=encoded_patch)
            nodes = nodes_to_numpy_ceasar(nodes=nodes, feature_dim=cfg.model.transformer_config.trans_dim, cfg=cfg)
            # prevent information/label leakage
            if cfg.shuffle_nodes:  
                np.random.shuffle(nodes)
            # allow max 1 edge per leaf node and parent
            hierarchy_edges = generate_edges(nodes=nodes)
            # save the graph:
            save_graph(nodes=nodes, hierarchy_edges=hierarchy_edges, path=graph_data_path / file_name)
    print(f'{len([p for p in graph_data_path.iterdir() if p.is_dir()])} graphs created')




