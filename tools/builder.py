import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, ReduceLROnPlateau

from models.build import build_model_from_cfg


def model_builder(cfg):
    model = build_model_from_cfg(cfg=cfg)
    return model


def build_optim_scheme(base_model, cfg):
    """
    Build an optimization scheme (optimizer, scheduler) based on the configurations from config. 
    This scheme is used in the configure_optimizers() method of the MAELightningModule.
    
    :param base_model: The model that is returned by model_builder(cfg.model)
    :param cfg: The configuration extracted from the configuration YAML file. 
    :returns: torch optimizer and torch lr scheduler. 
    """

    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    # get the optimizer
    opti_config = cfg.optimizer
    if opti_config.type == 'AdamW':
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    else:
        raise NotImplementedError(f"Optimizer: {opti_config}")
    
    # get the scheduler
    sche_config = cfg.scheduler
    if sche_config.type == 'CosLR':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=sche_config.kwargs.initial_epochs,
            T_mult=2,
            eta_min=1e-6)
    elif sche_config.type == 'ExpLR':
        scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=sche_config.kwargs.gamma
        )
    elif sche_config.type == 'RedPlatLR':
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='min', 
            factor=0.1, 
            patience=10, 
            threshold=0.0001
        )
    else:
        raise NotImplementedError()
    
    return optimizer, scheduler
    

def resume_model(base_model, args, logger=None):
    NotImplementedError()


def save_checkpoint(base_model, ckpt_path, logger=None):
    NotImplementedError()


def load_model(base_model, ckpt_path, logger=None):
    NotImplementedError()