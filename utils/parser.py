import os
import argparse
import logging
from faker import Faker
from pathlib import Path


def get_args() -> argparse.Namespace:
    """
    This function defines a base argument parses which we can use in or main file to handle the input
    arguments provided by the user. 
    """
    parser = argparse.ArgumentParser(description="Deep Learning Experiment Argument Parser")

    parser.add_argument('--config', type=str, help='YAML configuration file', required=True)
    parser.add_argument('--finetune', action='store_true', help="Finetune a pretrained model")
    parser.add_argument('--build_gnn_ds', action='store_true', help='Build a GNN dataset using PointMAE latentes')
    parser.add_argument('--build_default_graph', action='store_true', help='Build the default graph used by gnn ds builder')
    parser.add_argument('--train_gnn', action='store_true', help="train the GNN for node-level prediction")
    parser.add_argument('--train_mlp', action='store_true', help="train the MLP for part label prediction")
    parser.add_argument('--k_fold', action='store_true', help="Run k-fold cross validation")
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--wandb', action='store_true', help="Connect to Weights and Biases")
    parser.add_argument('--sweep', action='store_true', help="Run a Weights and Biases hyperparameter sweep")
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--random_seed', action='store_true', help='whether to set a random seed.')

    
    args = parser.parse_args()
    validate_args(args)
    configure_and_create_paths(args)

    return args


def validate_args(args: argparse.Namespace) -> None:
    """
    Performs basic validation on the parsed arguments.
    Raise ValueError if validation fails
    """
    if args.config and not Path(args.config).exists():
        raise ValueError(f'Config file {args.config} does not exist')


def configure_and_create_paths(args: argparse.Namespace) -> None:
    """
    Configure path arguments and create necessary paths
    """
    base_path = Path('./experiments')
    config_path = Path(args.config)
    args.experiment_path = base_path / config_path.stem / config_path.parent.stem / args.exp_name
    args.ckpnt_path = args.experiment_path / 'ckpnts'

    faker = Faker()
    # Generate three random words and combine them with an underscore
    random_name = f"{faker.word()}_{faker.word()}_{faker.word()}"
    args.log_name = random_name

    create_experiment_dir(args.experiment_path)
    create_experiment_dir(args.ckpnt_path)


def create_experiment_dir(path: Path) -> None:
    """
    Creates the specified directory path if it doesn't already exist.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f'Created path at {path}')
    except Exception as e:
        logging.error(f'Error creating path {path}: {e}')