import random
import torch
import hashlib
import numpy as np

MODEL_LAYERS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-14B": 40,
    "EleutherAI/pythia-70m": 6,
    "EleutherAI/pythia-160m": 12,
    "google/gemma-2-2b": 26,
    "google/gemma-2-9b": 42,
}

CONCEPT_CATEGORIES = {
    "sycophantic": {
        "base_path": "dataset/sycophantic.json",
        "dataset_key": "instruction",
        "loader_type": "single_file_with_pos_neg",
        "instruction_key": "instruction",
    },
    "evil": {
        "base_path": "dataset/evil.json",
        "dataset_key": "instruction",
        "loader_type": "single_file_with_pos_neg",
        "instruction_key": "instruction",
    },
    "optimistic": {
        "base_path": "dataset/optimistic.json",
        "dataset_key": "instruction",
        "loader_type": "single_file_with_pos_neg",
        "instruction_key": "instruction",
    },
    "language_en_fr_paired": {
        "base_path": "dataset/en_fr.json",
        "dataset_key": "instruction",
        "loader_type": "single_file_with_pos_neg",
        "instruction_key": "instruction",
    },
    "refusal": {
        "base_path": "dataset/refusal.json",
        "dataset_key": "instruction",
        "loader_type": "single_file_with_pos_neg",
        "instruction_key": "instruction",
    },
}


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
    raise AttributeError("Unable to locate transformer layers container on model")


