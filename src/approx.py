import argparse
import torch
import transformers
import os
import datasets
import gc
from loguru import logger
from utils import set_seed, _get_layers_container
from tqdm import tqdm

MODEL_LAYERS = {
    "google/gemma-3-270m-it": 18,
    "google/gemma-3-4b-it": 34,  # TODO: replace with Gemma2, and add Mistral3
    "google/gemma-3-12b-it": 48,
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8b": 36,
    "Qwen/Qwen3-14B": 40,
    "EleutherAI/pythia-70m": 6,
    "EleutherAI/pythia-410m": 24,
    "EleutherAI/pythia-160m": 12,
}

CONCEPT_CATEGORIES = {
    "safety": ["assets/harmbench", "instruction"],
    "language_en_fr": ["assets/language_translation", "text"],
    "random": ["assets/harmbench", "instruction"],
    "random1": ["assets/harmbench", "instruction"],
    "random2": ["assets/language_translation", "text"],
    "random3": ["assets/language_translation", "text"],
}


def config() -> argparse.Namespace:
    """
    Config for approximation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        choices=MODEL_LAYERS.keys(),
    )
    parser.add_argument(
        "--concept_category",
        type=str,
        default="safety",
        choices=CONCEPT_CATEGORIES.keys(),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approx_dataset_size", type=int, default=30)
    parser.add_argument("--alpha_factor", type=int, default=100)
    parser.add_argument("--alpha_min", type=float, default=1)
    return parser.parse_args()


def approx() -> None:
    """
    Test whether the approximation is correct
    """
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/approx.log")
    logger.info("Starting Approximation...")
    args = config()
    set_seed(args.seed)
    model_name = args.model.split("/")[-1]
    dtype = getattr(torch, args.dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map=args.device, dtype=dtype, trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, use_fast=True, dtype=dtype
    )
    max_layers = MODEL_LAYERS[args.model]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"args: {args}")

    for concept_category_name, concept_category_metadata in CONCEPT_CATEGORIES.items():
        save_path = f"assets/linearity/{model_name}/{concept_category_name}.pt"
        concept_vectors = torch.load(save_path)
        if concept_category_name in ["language_en_fr", "random2", "random3"]:
            dataset_path = os.path.join(concept_category_metadata[0], "en.jsonl")
            dataset = datasets.load_dataset(
                "json", data_files=dataset_path, split="train"
            )
            dataset_key = concept_category_metadata[1]
        elif concept_category_name in ["random1", "safety", "random"]:
            dataset_path = os.path.join(
                concept_category_metadata[0], "harmful_data.jsonl"
            )
            dataset = datasets.load_dataset(
                "json", data_files=dataset_path, split="train"
            )
            dataset_key = concept_category_metadata[1]
        else:
            raise ValueError(f"Invalid concept category name: {concept_category_name}")
        dataset = dataset.shuffle().select(range(args.approx_dataset_size))
        input_prompts = dataset[: args.approx_dataset_size][dataset_key]
        input_ids = tokenizer(
            input_prompts, return_tensors="pt", truncation=True, padding=True
        ).to(args.device)
        input_ids = input_ids.input_ids
        for layer_idx in tqdm(range(max_layers), desc="Approximation"):
            concept_vector = concept_vectors[layer_idx, :]
            logger.info(f"Concept vector: {concept_vector.shape}")
            layers_container = _get_layers_container(model)
            target_layer_module = layers_container[layer_idx]
            original_output = model(input_ids, output_hidden_states=True)
            last_layer_hidden_states = original_output.hidden_states[-1]
            dim_middle_states = last_layer_hidden_states.shape[-1]
            seq_len = last_layer_hidden_states.shape[1]
            
            for idx in tqdm(range(args.alpha_factor)):
                cur_approx_alpha = args.alpha_min * (idx + 1)
                
                def _forward_hook(module, inputs, output):
                    # Ensure concept vector matches device/dtype
                    if isinstance(output, tuple):
                        hidden = output[0]
                        vec = concept_vector.to(
                            device=hidden.device, dtype=hidden.dtype
                        )
                        hidden = hidden + cur_approx_alpha * vec
                        return (hidden,) + output[1:]
                    else:
                        vec = concept_vector.to(
                            device=output.device, dtype=output.dtype
                        )
                        return output + cur_approx_alpha * vec
                
                hook_handle = target_layer_module.register_forward_hook(
                    _forward_hook
                )
                steered_output = model(input_ids, output_hidden_states=True)
                steered_last_layer_hidden_states = steered_output.hidden_states[-1]
                hook_handle.remove()
                
                hidden_states_diff = steered_last_layer_hidden_states - last_layer_hidden_states
                # Jacobian = compute_Jacobian(hidden_states_diff, last_layer_hidden_states)


def compute_Jacobian(hidden_states_diff, last_layer_hidden_states):
    """
    Compute the Jacobian of the hidden states difference
    """
    ...

if __name__ == "__main__":
    approx()
