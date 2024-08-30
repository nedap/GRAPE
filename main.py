import torch
from pytorch_lightning.loggers import WandbLogger

from utils import parser
from utils.config import *
from utils.logger import *
from utils.misc import set_random_seed
from tools.model_trainer import pre_train, train_gnn, train_mlp
from tools.build_gnn_ds import build_gnn_ds_partnet, build_gnn_ds_ceasar
from datasets.graph.create_default_table_graph import build_default_table_graph
from datasets.graph.create_default_ceasar_graph import build_default_ceasar_graph



def main():
    # args
    args = parser.get_args()
    # Terminal Logger
    terminal_logger = get_logger(name=args.log_name)
    # config
    cfg = get_cfg(args=args, logger=terminal_logger)
    if args.wandb and not args.k_fold:
        wandb_logger = WandbLogger(
            name=args.log_name, 
            project=cfg.wandb_project, 
            config=cfg, 
            save_dir=args.experiment_path
        )
    else:
        wandb_logger = None

    # CUDA
    args.use_gpu = True if cfg.device.name == 'cuda' and torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = args.use_gpu
    # random seed 
    if args.random_seed : set_random_seed(logger=terminal_logger, seed=args.seed + args.local_rank, deterministic=args.deterministic)

    if args.train_gnn: # train the GNN for node level prediction
        train_gnn(args=args, cfg=cfg, wandb_logger=wandb_logger, terminal_logger=terminal_logger, k_fold=args.k_fold)
    elif args.build_gnn_ds: # build the GNN dataset using point-MAE  
        # build_gnn_ds(args=args, cfg=cfg, wandb_logger=wandb_logger, terminal_logger=terminal_logger) 
        build_gnn_ds_ceasar(args=args, cfg=cfg, wandb_logger=wandb_logger, terminal_logger=terminal_logger) 
    elif args.build_default_graph:
        build_default_ceasar_graph(args=args, cfg=cfg)
    elif args.train_mlp:
        train_mlp(args=args, cfg=cfg, wandb_logger=wandb_logger, k_fold=args.k_fold)
    else:
         # pretrain Point-MAE with self-supervised learning
        pre_train(args=args, cfg=cfg, wandb_logger=wandb_logger, terminal_logger=terminal_logger)
    print_log('END OF PROGRAM - Have a good day!', logger=terminal_logger)


if __name__ == '__main__':
    main()
