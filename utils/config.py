import yaml
import os
import argparse

from easydict import EasyDict
from typing import Optional


def merge_new_config(target_config: EasyDict, source_config: dict) -> EasyDict:
    """
    Merges the source configuration dictionary into the target configuration dictionary.
    
    :param target_config: The target configuration dictionary as an EasyDict.
    :param source_config: The source configuration dictionary as a regular dict.
    :return: The merged configuration dictionary as an EasyDict.
    """
    for key, value in source_config.items():
        # Handle the special '_base_' key
        if key == '_base_':
            base_config_path = value
            if not os.path.exists(base_config_path):
                raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")
            with open(base_config_path, 'r') as file:
                base_config = yaml.safe_load(file)
            merge_new_config(target_config, EasyDict(base_config))  # Convert to EasyDict
        elif isinstance(value, dict):
            # Convert value to EasyDict and ensure the key exists in the target config
            value_as_easydict = EasyDict(value)
            target_config.setdefault(key, value_as_easydict)
            # Recursively merge nested dictionaries
            merge_new_config(target_config[key], value_as_easydict)
        else:
            # Overwrite/insert the key-value pair in the target config
            target_config[key] = value

    return EasyDict(target_config)


def cfg_from_yaml_file(cfg_file: str) -> EasyDict:
    """
    Loads and returns a configuration from a YAML file.
    
    :param cfg_file: Path to the YAML configuration file.
    :return: Configuration as an EasyDict.
    """
    cfg = EasyDict()
    with open(cfg_file, 'r') as f:
        new_cfg = yaml.safe_load(f)
        cfg = merge_new_config(cfg, new_cfg)
    return cfg


def get_cfg(args: argparse.Namespace, logger: Optional[str] = None) -> EasyDict:
    """
    Retrieves the configuration either from a specified file or from a previous experiment.
    
    :param args: argparse Namespace containing the arguments.
    :param logger: Optional logger name.
    :return: Configuration as an EasyDict.
    """
    cfg = cfg_from_yaml_file(args.config)
    perform_cfg_checks(cfg=cfg)
    return cfg


def perform_cfg_checks(cfg: argparse.Namespace) -> None:
    """
    Check the values in the config.
    There are certain restraints that need to be followed to guarantee that the model is created as intended. 
    """

    if cfg.model.NAME == 'GNN' or 'MLP':
        pass
    else:
        # empirically found a ratio of numPatches × pointsPerPatch = 2N to be effective, check if this holds
        group_size = int(cfg.model.group_size)
        num_group = int(cfg.model.num_group)
        num_points = int(cfg.npoints)

        # if group_size * num_group!= 2 * num_points:
        #     raise ValueError(f"group_size * num_group != 2 * npoints\n ({group_size} * {num_group} != 2 * {num_points})")

        # Check for model_finetune and compare transformer_config if present
        if hasattr(cfg, 'model_finetune'):
            # Extract transformer configurations
            model_config = cfg.model.transformer_config
            finetune_config = cfg.model_finetune.transformer_config

            # Define keys to compare
            keys_to_compare = ['trans_dim', 'encoder_dims', 'depth', 'num_heads']

            # Check if all specified keys are equal in both configurations
            for key in keys_to_compare:
                if model_config.get(key, None) != finetune_config.get(key, None):
                    raise ValueError(
                        f"model and model_finetune have different {key} values - {model_config.get(key)} != {finetune_config.get(key)}")
