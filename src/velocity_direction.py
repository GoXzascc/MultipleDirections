# velocity_direction.py
# Compute cos(v(α), v(α-ε)) to see if velocity direction is changing
# where v(α) = h(α) - h(α-ε) is the "velocity" at alpha

import argparse
import torch
import transformers
import os
from utils import (
    MODEL_LAYERS,
    CONCEPT_CATEGORIES,
    set_seed,
    _get_layers_container,
    load_concept_datasets,
)
from loguru import logger
from tqdm import tqdm


def velocity_direction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=16)
    parser.add_argument("--alpha_min", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=1e7)
    parser.add_argument("--alpha_points", type=int, default=200)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/velocity_direction.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    model_name = args.model.split("/")[-1]
    os.makedirs(f"assets/velocity_direction/{model_name}", exist_ok=True)
    logger.info(f"Starting velocity direction measurement...")
    logger.info(f"Loading model: {args.model}")
    device = "cuda"
    dtype = torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map=device, dtype=dtype, trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, use_fast=True, device=device, dtype=dtype
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_layers = MODEL_LAYERS[args.model]

    for concept_category_name, concept_category_config in CONCEPT_CATEGORIES.items():
        concept_vectors = torch.load(
            f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
        )

        positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
            concept_category_name, concept_category_config
        )
        actual_size = min(args.test_size, len(positive_dataset))
        input_prompts = positive_dataset[:actual_size][dataset_key]
        input_ids = tokenizer(
            input_prompts, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        input_ids = input_ids.input_ids

        num_layers = min(max_layers, concept_vectors.shape[0])
        vector_dim = concept_vectors.shape[1]

        # Load or generate random vector (reuse from directional_change if exists)
        random_vector_dir = f"assets/directional_change/{model_name}/random_vectors"
        random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"

        if os.path.exists(random_vector_path):
            random_vector_data = torch.load(random_vector_path)
            random_vector = random_vector_data["random_vector"]
            logger.info(f"Using saved random vector for {concept_category_name}")
        else:
            # Generate a random vector with the same dimension as concept vectors
            os.makedirs(random_vector_dir, exist_ok=True)
            random_vector = torch.randn(vector_dim, dtype=torch.float32)
            random_vector = random_vector / random_vector.norm()
            torch.save(
                {
                    "random_vector": random_vector,
                    "concept_category": concept_category_name,
                },
                random_vector_path,
            )
            logger.info(f"Generated and saved random vector for {concept_category_name}")

        # Run velocity direction analysis with concept vector and random vector
        for vector_type, vector_source in [
            ("concept", concept_vectors),
            ("random", random_vector),
        ]:
            results: dict[int, dict[str, torch.Tensor]] = {}

            for layer_idx in tqdm(
                range(num_layers - 1),
                desc=f"Measuring velocity direction ({concept_category_name}, {vector_type})",
            ):
                if vector_type == "concept":
                    steering_vector = concept_vectors[layer_idx, :]
                else:
                    steering_vector = random_vector

                layer_results = compute_velocity_direction(
                    model=model,
                    input_ids=input_ids,
                    steering_vector=steering_vector,
                    layer_idx=layer_idx,
                    dtype=dtype,
                    device=device,
                    alpha_min=args.alpha_min,
                    alpha_max=args.alpha_max,
                    alpha_points=args.alpha_points,
                )

                results[layer_idx] = layer_results

            # Save results
            save_path = f"assets/velocity_direction/{model_name}/velocity_direction_{concept_category_name}_{vector_type}.pt"
            torch.save(
                {
                    "model": args.model,
                    "concept_category": concept_category_name,
                    "vector_type": vector_type,
                    "alpha_min": args.alpha_min,
                    "alpha_max": args.alpha_max,
                    "alpha_points": args.alpha_points,
                    "results": results,
                },
                save_path,
            )
            logger.info(f"Saved velocity direction ({vector_type}) to {save_path}")


def compute_velocity_direction(
    model,
    input_ids,
    steering_vector: torch.Tensor,
    layer_idx: int,
    dtype,
    device,
    alpha_min: float = 1e-3,
    alpha_max: float = 1e7,
    alpha_points: int = 200,
) -> dict[str, torch.Tensor]:
    """
    Compute velocity direction consistency: cos(v(α), v(α-ε))
    
    where v(α) = h(α) - h(α-ε) is the "velocity" (direction of change) at alpha.
    
    This measures whether the trajectory is curving in hidden space:
    - cos ≈ 1: velocity direction is stable (straight trajectory)
    - cos < 1: velocity direction is changing (curved trajectory)
    
    Returns:
        Dictionary containing:
        - alpha: tensor of alpha values (excluding first point)
        - cos_velocity: cos(v(α_i), v(α_{i-1})) for consecutive alpha pairs
    """
    # Build alpha sweep in log-space
    alphas = torch.logspace(
        float(torch.log10(torch.tensor(alpha_min))),
        float(torch.log10(torch.tensor(alpha_max))),
        steps=int(alpha_points),
    ).tolist()

    layers_container = _get_layers_container(model)
    target_layer_module = layers_container[layer_idx]
    last_layer_module = layers_container[len(layers_container) - 1]

    def _run_with_alpha(alpha_value: float) -> torch.Tensor:
        """Run model with steering and capture last layer hidden states."""
        captured: dict[str, torch.Tensor] = {}

        def _last_layer_forward_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden.detach()
            return output

        def _steer_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                vec = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
                hidden = hidden + (alpha_value * vec)
                return (hidden,) + output[1:]
            vec = steering_vector.to(device=output.device, dtype=output.dtype)
            return output + (alpha_value * vec)

        last_handle = last_layer_module.register_forward_hook(_last_layer_forward_hook)
        steer_handle = target_layer_module.register_forward_hook(_steer_hook)
        _ = model(input_ids, output_hidden_states=True)
        steer_handle.remove()
        last_handle.remove()

        h = captured.get("h", None)
        if h is None:
            raise RuntimeError("Failed to capture hidden states")
        return h

    def _hidden_to_flat(h: torch.Tensor) -> torch.Tensor:
        """Flatten hidden states: [batch, seq, d] -> [batch*seq, d]"""
        hs_dim = h.shape[-1]
        return h.reshape(-1, hs_dim).to(torch.bfloat16)

    def _cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute average cosine similarity between batch of vectors.
        
        Args:
            a: tensor of shape [N, d]
            b: tensor of shape [N, d]
        
        Returns:
            Average cosine similarity (scalar)
        """
        a_norm = torch.nn.functional.normalize(a, dim=-1)
        b_norm = torch.nn.functional.normalize(b, dim=-1)
        cos_sim = (a_norm * b_norm).sum(dim=-1)  # [N]
        return cos_sim.mean().item()

    # Compute h(α) for all alpha values first
    h_list = []
    for alpha in tqdm(alphas, desc=f"  Layer {layer_idx} (collecting h)", leave=False):
        h_alpha = _hidden_to_flat(_run_with_alpha(alpha))
        h_list.append(h_alpha)

    # Compute velocity: v(α_i) = h(α_i) - h(α_{i-1})
    # Then compute cos(v(α_i), v(α_{i-1}))
    alpha_out = []
    cos_velocity_list = []

    for i in tqdm(range(2, len(alphas)), desc=f"  Layer {layer_idx} (computing cos)", leave=False):
        # v(α_i) = h(α_i) - h(α_{i-1})
        v_curr = h_list[i] - h_list[i - 1]
        # v(α_{i-1}) = h(α_{i-1}) - h(α_{i-2})
        v_prev = h_list[i - 1] - h_list[i - 2]
        
        # cos(v(α_i), v(α_{i-1}))
        cos_v = _cosine_similarity_batch(v_curr, v_prev)
        
        alpha_out.append(alphas[i])
        cos_velocity_list.append(cos_v)

    return {
        "alpha": torch.tensor(alpha_out, dtype=torch.float32),
        "cos_velocity": torch.tensor(cos_velocity_list, dtype=torch.float32),
    }


if __name__ == "__main__":
    velocity_direction()
