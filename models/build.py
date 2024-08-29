from utils import registry


MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a model, defined by NAME key in the config file.

    :param cfg: Configuration of the model based on variables in config file. 
    :returns: A constructed model specified by the NAME key in the cfg dictionary.
    """
    return MODELS.build(cfg, **kwargs)
