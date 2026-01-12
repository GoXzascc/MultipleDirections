import random
import torch
import hashlib
import os
import gc
import transformer_lens
import datasets
import numpy as np
from tqdm import tqdm
from loguru import logger

MODEL_LAYERS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-14B": 40,
    "EleutherAI/pythia-70m": 6,
    "EleutherAI/pythia-160m": 12,
    "google/gemma-2-2b": 26,
    "google/gemma-2-9b": 42,
    "Qwen/Qwen3-0.6B": 27,
}

CONCEPT_CATEGORIES = {
    "safety": {
        "base_path": "dataset/harmbench",
        "dataset_key": "instruction",
        "loader_type": "separate_files",
        "positive_file": "harmful_data.jsonl",
        "negative_file": "harmless_data.jsonl",
    },
    "language_en_fr": {
        "base_path": "dataset/language_translation",
        "dataset_key": "text",
        "loader_type": "separate_files",
        "positive_file": "en.jsonl",
        "negative_file": "fr.jsonl",
    },
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


def _load_separate_files_dataset(
    base_path: str, positive_file: str, negative_file: str
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load datasets from separate positive and negative files.
    
    Args:
        base_path: Base directory path containing the files
        positive_file: Name of the positive examples file
        negative_file: Name of the negative examples file
    
    Returns:
        Tuple of (positive_dataset, negative_dataset)
    """
    positive_dataset_path = os.path.join(base_path, positive_file)
    negative_dataset_path = os.path.join(base_path, negative_file)
    
    positive_dataset = datasets.load_dataset(
        "json", data_files=positive_dataset_path, split="train"
    )
    negative_dataset = datasets.load_dataset(
        "json", data_files=negative_dataset_path, split="train"
    )
    
    return positive_dataset, negative_dataset


def _load_single_file_with_pos_neg(
    file_path: str, instruction_key: str, dataset_key: str
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Load datasets from a single file containing positive and negative examples.
    
    Args:
        file_path: Path to the JSON file
        instruction_key: Key in the JSON file containing the instruction array
        dataset_key: Key name to use when creating the dataset
    
    Returns:
        Tuple of (positive_dataset, negative_dataset)
    """
    dataset_file = datasets.load_dataset(
        "json", data_files=file_path, split="train"
    )
    
    # Extract positive and negative instructions
    instructions = dataset_file[0][instruction_key]
    positive_prompts = [item["pos"] for item in instructions]
    negative_prompts = [item["neg"] for item in instructions]
    
    # Create datasets from the prompts
    positive_dataset = datasets.Dataset.from_dict({dataset_key: positive_prompts})
    negative_dataset = datasets.Dataset.from_dict({dataset_key: negative_prompts})
    
    return positive_dataset, negative_dataset


def load_concept_datasets(
    concept_category_name: str, concept_category_config: dict
) -> tuple[datasets.Dataset, datasets.Dataset, str]:
    """Load positive and negative datasets for a concept category.
    
    Args:
        concept_category_name: Name of the concept category
        concept_category_config: Configuration dictionary for the concept category
    
    Returns:
        Tuple of (positive_dataset, negative_dataset, dataset_key)
    """
    loader_type = concept_category_config["loader_type"]
    dataset_key = concept_category_config["dataset_key"]
    
    if loader_type == "separate_files":
        base_path = concept_category_config["base_path"]
        positive_file = concept_category_config["positive_file"]
        negative_file = concept_category_config["negative_file"]
        positive_dataset, negative_dataset = _load_separate_files_dataset(
            base_path, positive_file, negative_file
        )
    elif loader_type == "single_file_with_pos_neg":
        file_path = concept_category_config["base_path"]
        instruction_key = concept_category_config["instruction_key"]
        positive_dataset, negative_dataset = _load_single_file_with_pos_neg(
            file_path, instruction_key, dataset_key
        )
    else:
        raise ValueError(
            f"Unknown loader_type '{loader_type}' for concept category '{concept_category_name}'"
        )
    
    return positive_dataset, negative_dataset, dataset_key


def concept_vector(
    max_dataset_size: int = 300
):
    """used to obtain the concept vector for the given concept category
    Args:
        model: the model to obtain the concept vector
        max_dataset_size: the maximum size of the dataset to obtain the concept vector
    Returns:
        concept_vector: the concept vector
    """
    logger.info("Obtaining concept vectors...")
    os.makedirs("assets/concept_vectors", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/concept_vectors.log")
    device = "cuda"
    dtype = torch.bfloat16
    for model_name, max_layers in MODEL_LAYERS.items():
        model = transformer_lens.HookedTransformer.from_pretrained(
            model_name, device=device, dtype=dtype, trust_remote_code=True
        )
        for (
            concept_category_name,
            concept_category_config,
        ) in CONCEPT_CATEGORIES.items():
            concept_vector = obtain_concept_vector(
                model,
                max_layers,
                concept_category_name,
                concept_category_config,
                max_dataset_size,
                model_name,
            )

def obtain_concept_vector(
    model: transformer_lens.HookedTransformer,
    max_layers: int,
    concept_category_name: str,
    concept_category_config: dict,
    max_dataset_size: int = 300,
    model_name: str = "Qwen/Qwen3-1.7B",
    methods: str = "difference-in-means",
) -> torch.Tensor:
    """Obtain concept vector for a given concept category.
    
    Args:
        model: The model to obtain the concept vector from
        max_layers: Maximum number of layers to process
        concept_category_name: Name of the concept category
        concept_category_config: Configuration dictionary for the concept category
        max_dataset_size: Maximum size of the dataset to use
        model_name: Name of the model
        methods: Method to use for obtaining concept vectors
    
    Returns:
        Tuple of (concept_vector, positive_dataset, dataset_key)
    """
    # Load datasets using the unified loading system
    positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
        concept_category_name, concept_category_config
    )
    
    # Set up save path
    os.makedirs(f"assets/concept_vectors/{model_name}", exist_ok=True)
    save_path = f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"

    # Get concept vectors
    concept_vector = get_concept_vectors(
        model=model,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        layer=max_layers,
        device="cuda",
        positive_dataset_key=dataset_key,
        negative_dataset_key=dataset_key,
        methods=methods,
        save_path=save_path,
        max_dataset_size=max_dataset_size,
    )
    torch.cuda.empty_cache()
    gc.collect()

    return concept_vector, positive_dataset, dataset_key


def get_concept_vectors(
    model: transformer_lens.HookedTransformer,
    positive_dataset: datasets.Dataset,
    negative_dataset: datasets.Dataset,
    layer: int,
    device: str,
    positive_dataset_key: str,
    negative_dataset_key: str,
    methods: str,
    save_path: str,
    max_dataset_size: int = 300,
) -> torch.Tensor:
    """used to get the concept vectors using the specified method

    Args:
        model (transformer_lens.HookedTransformer): the model to get the concept vectors
        positive_dataset (datasets.Dataset): the positive dataset to get the concept vectors
        negative_dataset (datasets.Dataset): the negative dataset to get the concept vectors
        layer (int): the layer to get the concept vectors
        device (str): the device to get the concept vectors
        positive_dataset_key (str): the key of the positive dataset to get the concept vectors
        negative_dataset_key (str): the key of the negative dataset to get the concept vectors
        save_path (str): the path to save the concept vectors
        methods (str): the method to get the concept vectors
        max_dataset_size (int, optional): the maximum size of the dataset to get the concept vectors. Defaults to 300.

    Returns:
        torch.Tensor: the concept vectors
    """
    # TODO: more methods
    if methods == "difference-in-means":
        difference_in_means = DifferenceInMeans(
            model,
            positive_dataset,
            negative_dataset,
            layer=layer,
            device=device,
            positive_dataset_key=positive_dataset_key,
            negative_dataset_key=negative_dataset_key,
            max_dataset_size=max_dataset_size,
        )
        concept_vector = difference_in_means.get_concept_vectors(
            save_path=save_path,
            is_save=True,
        )
    else:
        raise ValueError(f"Invalid method: {methods}")
    return concept_vector


class DifferenceInMeans:
    def __init__(
        self,
        model: transformer_lens.HookedTransformer,
        positive_dataset: datasets.Dataset,
        negative_dataset: datasets.Dataset,
        layer: int,
        device: str,
        positive_dataset_key: str,
        negative_dataset_key: str,
        max_dataset_size: int = 300,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """used to calculate the concept vector using difference-in-means

        Args:
            model (transformer_lens.HookedTransformer): the model to calculate the concept vector
            positive_dataset (datasets.Dataset): the positive dataset to calculate the concept vector
            negative_dataset (datasets.Dataset): the negative dataset to calculate the concept vector
            layer (int): the layer to calculate the concept vector
            device (str): the device to calculate the concept vector
            positive_dataset_key (str): the key of the positive dataset to calculate the concept vector
            negative_dataset_key (str): the key of the negative dataset to calculate the concept vector
            max_dataset_size (int, optional): the maximum size of the dataset to calculate the concept vector. Defaults to 300.
        """
        self.model = model
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset
        self.layers = list(range(layer))
        self.device = device
        self.positive_dataset_key = positive_dataset_key
        self.negative_dataset_key = negative_dataset_key
        self.max_dataset_size = max_dataset_size
        self.dtype = dtype

    def get_concept_vectors(
        self, save_path: str, is_save: bool = False
    ) -> torch.Tensor:
        """used to calculate the concept vectors using difference-in-means

        Args:
            save_path (str): the path to save the concept vectors
            is_save (bool, optional): whether to save the concept vectors. Defaults to False.

        Returns:
            torch.Tensor: the concept vectors
        """
        model_dimension = self.model.cfg.d_model
        layer_length = len(self.layers)
        logger.info(f"layer_length: {layer_length}")
        positive_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )
        negative_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )
        positive_token_length = 0
        negative_token_length = 0
        positive_dataset_size = (
            len(self.positive_dataset)
            if self.max_dataset_size > len(self.positive_dataset)
            else self.max_dataset_size
        )
        for i, example in tqdm(
            enumerate(self.positive_dataset), total=positive_dataset_size
        ):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.positive_dataset_key]
            _, positive_cache = self.model.run_with_cache(context)
            for layer in self.layers:
                positive_hidden_state = positive_cache[
                    f"blocks.{layer}.hook_resid_post"
                ].reshape(-1, model_dimension)
                positive_concept_vector[layer] += positive_hidden_state.sum(dim=0)
                if layer == 0:
                    current_token_length = positive_hidden_state.shape[0]
                    positive_token_length += current_token_length
        negative_dataset_size = (
            len(self.negative_dataset)
            if self.max_dataset_size > len(self.negative_dataset)
            else self.max_dataset_size
        )
        for i, example in tqdm(
            enumerate(self.negative_dataset), total=negative_dataset_size
        ):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.negative_dataset_key]
            _, negative_cache = self.model.run_with_cache(
                context, stop_at_layer=layer + 1
            )
            for layer in self.layers:
                negative_hidden_state = negative_cache[
                    f"blocks.{layer}.hook_resid_post"
                ].reshape(-1, model_dimension)
                negative_concept_vector[layer] += negative_hidden_state.sum(dim=0)
                if layer == 0:
                    current_token_length = negative_hidden_state.shape[0]
                    negative_token_length += current_token_length
        positive_concept_vector /= positive_token_length
        negative_concept_vector /= negative_token_length
        concept_diff = positive_concept_vector - negative_concept_vector
        concept_diff = torch.nn.functional.normalize(concept_diff, dim=1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(concept_diff, save_path)
        logger.info(f"Concept vector shape: {concept_diff.shape}")
        logger.info(f"Concept vector: {concept_diff.norm(dim=1)}")
        return concept_diff


if __name__ == "__main__":
    concept_vector()
