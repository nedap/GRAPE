from typing import Optional, Dict, Any
from easydict import EasyDict as edict
from utils import registry


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
    """
    Build a dataset, defined by the 'NAME' key in the cfg dictionary.

    :param cfg: A dictionary containing the configuration for the dataset.
    :param default_args: Optional default arguments for building the dataset.
    :return: A constructed dataset specified by the NAME key in the cfg dictionary.
    """
    return DATASETS.build(cfg, default_args=default_args)


def build_dataset_from_path(cfg: Dict[str, Any], data_path: str, default_args: Optional[Dict[str, Any]] = None) -> Any:
    """
    Build a dataset, defined by the 'NAME' key in the cfg dictionary,
    with an option to specify a data path.

    :param cfg: A dictionary containing the configuration for the dataset.
    :param data_path: Path to the folder that contains the data.
    :param default_args: Optional default arguments for building the dataset.
    :return: A constructed dataset specified by the NAME key in the cfg dictionary.
    """
    cfg = edict(cfg.copy())
    cfg.DATA_PATH = data_path
    return build_dataset_from_cfg(cfg, default_args=default_args)




