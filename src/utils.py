import random
import torch
import hashlib
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set the seed for the random number generator
    Args:
        seed: the seed to use
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None

def seed_from_name(name: str) -> int:
    """
    Get a seed from a string
    Args:
        name: the string to get a seed from
    Returns:
        the seed
    """
    h = hashlib.md5(name.encode()).hexdigest()
    return int(h, 16) % (2**31)


def _get_layers_container(hf_model):
    """used to get the layers container of the model
    Args:
        hf_model (transformers.PreTrainedModel): the model to get the layers container
    Returns:
        layers (list): the layers container of the model
    """
    # Common containers across HF architectures
    candidates = [
        (hf_model, "gpt_neox", "layers"),
        (hf_model, "model", "layers"),
        (hf_model, "transformer", "layers"),
        (hf_model, "transformer", "h"),
    ]
    for root_obj, root_attr, layers_attr in candidates:
        root = getattr(root_obj, root_attr, None)
        if root is None:
            continue
        layers = getattr(root, layers_attr, None)
        if layers is not None:
            return layers
    raise AttributeError(
        "Unable to locate transformer layers container on model"
    )