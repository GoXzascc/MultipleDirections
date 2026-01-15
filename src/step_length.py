import argparse
import torch
import transformers
import os
import random
from utils import (
    MODEL_LAYERS,
    CONCEPT_CATEGORIES,
    set_seed,
    _get_layers_container,
)
from extract_concepts import load_concept_datasets
from loguru import logger
from tqdm import tqdm


def step_length():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=16)
    parser.add_argument("--alpha_min", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=1e7)
    parser.add_argument("--alpha_points", type=int, default=200)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/step_length.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    model_name = args.model.split("/")[-1]
    os.makedirs(f"assets/step_length/{model_name}", exist_ok=True)
    logger.info(f"Starting step length measurement...")
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

        # Generate and save a random vector for this concept (reuse if exists)
        random_vector_dir = f"assets/step_length/{model_name}/random_vectors"
        os.makedirs(random_vector_dir, exist_ok=True)
        random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"

        if os.path.exists(random_vector_path):
            random_vector_data = torch.load(random_vector_path)
            random_vector = random_vector_data["random_vector"]
            logger.info(f"Using saved random vector for {concept_category_name}")
        else:
            # Generate a random vector with the same dimension as concept vectors
            random_vector = torch.randn(vector_dim, dtype=torch.float32)
            # Normalize the random vector to have unit norm
            random_vector = random_vector / random_vector.norm()
            torch.save(
                {
                    "random_vector": random_vector,
                    "concept_category": concept_category_name,
                },
                random_vector_path,
            )
            logger.info(
                f"Generated and saved random vector for {concept_category_name}"
            )

        # Run norm decomposition with concept vector and random vector separately
        for vector_type, vector_source in [
            ("concept", concept_vectors),
            ("random", random_vector),
        ]:
            results: dict[int, dict[str, torch.Tensor]] = {}

            for layer_idx in tqdm(
                range(num_layers - 1),
                desc=f"Measuring step length ({concept_category_name}, {vector_type})",
            ):
                if vector_type == "concept":
                    # Use the concept vector from the corresponding layer
                    steering_vector = concept_vectors[layer_idx, :]
                else:
                    # Use the same random vector for all layers
                    steering_vector = random_vector

                layer_results = compute_norm_decomposition(
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
            save_path = f"assets/step_length/{model_name}/norm_decomposition_{concept_category_name}_{vector_type}.pt"
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
            logger.info(f"Saved norm decomposition ({vector_type}) to {save_path}")


def compute_norm_decomposition(
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
    Compute norm decomposition for different alpha values.
    
    For each alpha, we compute:
    - h(alpha): hidden state after steering with alpha * steering_vector
    - h(0): hidden state without steering (baseline)
    - residual = h(alpha) - alpha * steering_vector
    
    Then decompose residual into:
    - parallel component: projection onto steering_vector direction
    - orthogonal component: component perpendicular to steering_vector
    
    Returns:
        Dictionary containing:
        - alpha: tensor of alpha values
        
        Residual A (no intercept): residual_A = h(α) - αv
        - total_norm_A: average per-token ||h(α) - αv||
        - parallel_norm_A: average per-token ||projection onto v||
        - ortho_norm_A: average per-token ||orthogonal component||
        
        Residual B (with intercept): residual_B = (h(α) - h(0)) - αv
        - total_norm_B: average per-token ||(h(α) - h(0)) - αv||
        - parallel_norm_B: average per-token ||projection onto v||
        - ortho_norm_B: average per-token ||orthogonal component||
        
        - h0_norm: average per-token ||h(0)|| (baseline hidden state norm)
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
            raise RuntimeError(
                "Failed to capture hidden states for norm computation"
            )
        return h

    def _hidden_to_flat(h: torch.Tensor) -> torch.Tensor:
        """Flatten hidden states: [batch, seq, d] -> [batch*seq, d]"""
        hs_dim = h.shape[-1]
        return h.reshape(-1, hs_dim).to(torch.bfloat16)

    # Normalize steering vector for projection computations
    steering_vec_bf16 = steering_vector.to(dtype=torch.bfloat16, device=device)
    steering_vec_normalized = steering_vec_bf16 / steering_vec_bf16.norm()

    # Get baseline hidden state (alpha = 0)
    h0 = _hidden_to_flat(_run_with_alpha(0.0))  # [batch*seq, d]
    h0_norm = torch.norm(h0, p=2, dim=1).mean().item()  # average per-token norm

    alpha_list = []
    # residual_A = h(α) - αv (no intercept, checking if h passes through origin)
    total_norm_A_list = []
    parallel_norm_A_list = []
    ortho_norm_A_list = []
    # residual_B = (h(α) - h(0)) - αv (with intercept, checking if h(α) ≈ h(0) + αv)
    total_norm_B_list = []
    parallel_norm_B_list = []
    ortho_norm_B_list = []

    for alpha in tqdm(alphas, desc=f"  Layer {layer_idx}", leave=False):
        # Get hidden state with steering
        h_alpha = _hidden_to_flat(_run_with_alpha(alpha))  # [batch*seq, d]
        
        # Compute steering contribution
        steering_contribution = alpha * steering_vec_bf16.unsqueeze(0)  # [1, d]
        
        # ===== Residual A: h(α) - αv (no intercept) =====
        residual_A = h_alpha - steering_contribution  # [batch*seq, d]
        
        # Project residual_A onto steering direction
        dot_A = torch.matmul(residual_A, steering_vec_normalized)  # [batch*seq]
        parallel_A = dot_A.unsqueeze(1) * steering_vec_normalized.unsqueeze(0)  # [batch*seq, d]
        ortho_A = residual_A - parallel_A  # [batch*seq, d]
        
        # Compute average norms for residual_A
        total_norm_A = torch.norm(residual_A, p=2, dim=1).mean().item()
        parallel_norm_A = torch.norm(parallel_A, p=2, dim=1).mean().item()
        ortho_norm_A = torch.norm(ortho_A, p=2, dim=1).mean().item()
        
        # ===== Residual B: (h(α) - h(0)) - αv (with intercept) =====
        hs_diff = h_alpha - h0  # [batch*seq, d]
        residual_B = hs_diff - steering_contribution  # [batch*seq, d]
        
        # Project residual_B onto steering direction
        dot_B = torch.matmul(residual_B, steering_vec_normalized)  # [batch*seq]
        parallel_B = dot_B.unsqueeze(1) * steering_vec_normalized.unsqueeze(0)  # [batch*seq, d]
        ortho_B = residual_B - parallel_B  # [batch*seq, d]
        
        # Compute average norms for residual_B
        total_norm_B = torch.norm(residual_B, p=2, dim=1).mean().item()
        parallel_norm_B = torch.norm(parallel_B, p=2, dim=1).mean().item()
        ortho_norm_B = torch.norm(ortho_B, p=2, dim=1).mean().item()
        
        alpha_list.append(alpha)
        total_norm_A_list.append(total_norm_A)
        parallel_norm_A_list.append(parallel_norm_A)
        ortho_norm_A_list.append(ortho_norm_A)
        total_norm_B_list.append(total_norm_B)
        parallel_norm_B_list.append(parallel_norm_B)
        ortho_norm_B_list.append(ortho_norm_B)

    return {
        "alpha": torch.tensor(alpha_list, dtype=torch.float32),
        # Residual A: h(α) - αv (no intercept)
        "total_norm_A": torch.tensor(total_norm_A_list, dtype=torch.float32),
        "parallel_norm_A": torch.tensor(parallel_norm_A_list, dtype=torch.float32),
        "ortho_norm_A": torch.tensor(ortho_norm_A_list, dtype=torch.float32),
        # Residual B: (h(α) - h(0)) - αv (with intercept)
        "total_norm_B": torch.tensor(total_norm_B_list, dtype=torch.float32),
        "parallel_norm_B": torch.tensor(parallel_norm_B_list, dtype=torch.float32),
        "ortho_norm_B": torch.tensor(ortho_norm_B_list, dtype=torch.float32),
        "h0_norm": torch.tensor(h0_norm, dtype=torch.float32),
    }


if __name__ == "__main__":
    step_length()
