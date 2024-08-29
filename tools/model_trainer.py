import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from datasets.pytorch_lightning import PartNetDataModule, GNNDataModule, PartNetEmbeddingsDataModule, CeasarDataModule
from models.pytorch_lightning import MAELightningModule, GNNLightningModule, MLPLightningModule
from utils.misc import run_dummy_classifier

# import to register the class
from models.pytorch_models import GNN, Point_MAE, MLP
from datasets.pytorch import PartNet, GraphDataset



def pre_train(args, cfg, wandb_logger, terminal_logger):
    """
    Pre-train the model. During this stage, a masked autoencoder is trained to reconstruct point clouds, 
    where a large portion of the point cloud has been masked. This encoder will serve as a pretrained backbone 
    for a downstream classification task. 

    :param args: The command line arguments. 
    :param config: The configuration, extracted from the configurtion YAML file.
    :param wandb_logger: Weights and Biases logger. 
    :param terminal_logger: A custom logger that only loggs to the terminal.
    """
    model = MAELightningModule(cfg=cfg, args=args)
    
    # data_module = PartNetDataModule(args=args, cfg=cfg) 
    data_module = CeasarDataModule(args=args, cfg=cfg) 

    callbacks = [
        ModelCheckpoint(monitor="Validation Loss", dirpath=args.ckpnt_path, filename=f'{args.log_name}-{{epoch:02d}}')
    ]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[int(cfg.device.device_id)], 
        max_epochs=cfg.max_epoch,
        logger=wandb_logger, 
        callbacks=callbacks,
        default_root_dir=args.experiment_path,
        log_every_n_steps=1,
    )
    # fit the model
    trainer.fit(model=model, datamodule=data_module)


def train_gnn(args, cfg, wandb_logger, terminal_logger, k_fold: bool = False):
    data_module = GNNDataModule(args=args, cfg=cfg, k_fold=k_fold) 
    if k_fold:
        for fold in range(5):
            print(f"Initiating fold {fold}")
            data_module.setup_kfold(fold=fold)
            
            # Initialize a new wandb run for each fold
            fold_run = wandb.init(
                project=cfg.wandb_project if not args.sweep else "GNN-arch-sweep",
                name=f"{args.log_name}_fold_{fold+1}",
                config=cfg,
                dir=args.experiment_path
            )
            fold_wandb_logger = WandbLogger(experiment=fold_run)
        
            # run the dummy classifier instead of actual model
            if getattr(cfg, 'dummy_classifier', False):
                print("Running Dummy Classifier.")
                run_dummy_classifier(data_module, cfg)
                fold_run.finish()
            # run the real model
            else:
                
                callbacks = [
                    EarlyStopping(monitor='Validation Loss', patience=8, verbose=True, mode='min'),
                    ModelCheckpoint(save_weights_only=True, mode="max", monitor="Validation Accuracy")
                ]
                
                trainer = pl.Trainer(
                    accelerator='gpu', 
                    devices=[int(cfg.device.device_id)],
                    max_epochs=cfg.epochs,
                    logger=fold_wandb_logger,
                    enable_progress_bar=True,
                    default_root_dir=args.experiment_path,
                    callbacks=callbacks, 
                    log_every_n_steps=1
                )
                model = GNNLightningModule(cfg, args)
                # model.logger = fold_wandb_logger  # Ensure the model has the correct logger
                trainer.fit(model, data_module)
                trainer.test(datamodule=data_module)
                fold_run.finish()  # Finish the wandb run for the current fold
    else:
        callbacks = [
            EarlyStopping(monitor='Validation Loss', patience=8, verbose=True, mode='min'),
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="Validation Accuracy")
        ]
        
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=[int(cfg.device.device_id)],
            max_epochs=cfg.epochs,
            logger=wandb_logger,
            enable_progress_bar=True,
            default_root_dir=args.experiment_path,
            callbacks=callbacks, 
            log_every_n_steps=1
        )
        model = GNNLightningModule(cfg, args)
        # model.logger = wandb_logger  # Ensure the model has the correct logger
        trainer.fit(model, data_module)
        trainer.test(datamodule=data_module)


def train_mlp(args, cfg, wandb_logger, k_fold: bool = False):
    data_module = PartNetEmbeddingsDataModule(args=args, cfg=cfg, k_fold=k_fold) 
    if k_fold:
        for fold in range(5):
            print(f"Initiating fold {fold}")
            data_module.setup_kfold(fold=fold)

            # Initialize a new wandb run for each fold
            fold_run = wandb.init(
                project=cfg.wandb_project if not args.sweep else "GNN-arch-sweep",
                name=f"{args.log_name}_fold_{fold+1}",
                config=cfg,
                dir=args.experiment_path
            )
            fold_wandb_logger = WandbLogger(experiment=fold_run)

            callbacks = [
                EarlyStopping(monitor='Validation Loss', patience=12, verbose=True, mode='min'),
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="Validation Accuracy")
            ]
        
            model = MLPLightningModule(cfg, args)

            trainer = pl.Trainer(
                accelerator='gpu', 
                devices=[int(cfg.device.device_id)],
                max_epochs=cfg.epochs,
                logger=fold_wandb_logger,
                enable_progress_bar=True,
                default_root_dir=args.experiment_path,
                callbacks=callbacks, 
                log_every_n_steps=1
            )

            trainer.fit(model, data_module)
            trainer.test(datamodule=data_module)
            fold_run.finish()
    else:
        callbacks = [
            EarlyStopping(monitor='Validation Loss', patience=12, verbose=True, mode='min'),
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="Validation Accuracy")
        ]
        
        trainer = pl.Trainer(
            accelerator='gpu', 
            devices=[int(cfg.device.device_id)],
            max_epochs=cfg.epochs,
            logger=wandb_logger,
            enable_progress_bar=True,
            default_root_dir=args.experiment_path,
            callbacks=callbacks, 
            log_every_n_steps=1
        )
        model = MLPLightningModule(cfg, args)
        trainer.fit(model, data_module)
        trainer.test(datamodule=data_module)
