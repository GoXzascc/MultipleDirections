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
    run_model_with_steering,
    hidden_to_flat,
    get_model_name_for_path,
    parse_layers_to_run,
)
from extract_concepts import load_concept_datasets
from loguru import logger
from tqdm import tqdm


def velocity_direction():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model name to process. If not specified, process all models. Available: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=int, default=16)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to use from the dataset",
    )
    parser.add_argument("--alpha_min", type=float, default=1e-3)
    parser.add_argument("--alpha_max", type=float, default=1e7)
    parser.add_argument("--alpha_points", type=int, default=200)
    parser.add_argument("--eps_rel", type=float, default=1e-2)
    parser.add_argument("--eps_min", type=float, default=1e-5)
    parser.add_argument("--eps_max", type=float, default=1.0)
    parser.add_argument(
        "--layers",
        type=str,
        default="25,50,75",
        help="Comma-separated percentages (e.g., '25,50,75') or layer indices (e.g., '5,10,15'). Use 'all' to run all layers.",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/velocity_direction.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    
    # Determine which models to process
    models_to_process = (
        [(args.model, MODEL_LAYERS[args.model])]
        if args.model is not None
        else list(MODEL_LAYERS.items())
    )
    
    logger.info(f"Starting velocity direction measurement...")
    logger.info(f"Models to process: {[model_name for model_name, _ in models_to_process]}")
    device = "cuda"
    dtype = torch.bfloat16
    
    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        model_name = get_model_name_for_path(model_full_name)
        os.makedirs(f"assets/velocity_direction/{model_name}", exist_ok=True)
        
        logger.info(f"Loading model: {model_full_name}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_full_name, device_map=device, dtype=dtype, trust_remote_code=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_full_name, use_fast=True, device=device, dtype=dtype
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Parse layers argument
        layers_to_run = parse_layers_to_run(args.layers, max_layers)
        
        logger.info(f"Total layers in model: {max_layers}")
        logger.info(f"Layers to run: {layers_to_run}")

        for concept_category_name, concept_category_config in CONCEPT_CATEGORIES.items():
            concept_vectors = torch.load(
                f"assets/concept_vectors/{model_name}/{concept_category_name}.pt"
            )

            positive_dataset, negative_dataset, dataset_key = load_concept_datasets(
                concept_category_name, concept_category_config
            )
            
            # Select prompts to meet max_tokens constraint
            selected_prompts = []
            total_tokens = 0
            for i in range(min(args.test_size, len(positive_dataset))):
                prompt = positive_dataset[i][dataset_key]
                # Tokenize to check length
                tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
                prompt_length = tokens.input_ids.shape[1]
                
                # Check if adding this prompt would exceed max_tokens
                if total_tokens + prompt_length > args.max_tokens and len(selected_prompts) > 0:
                    break
                
                selected_prompts.append(prompt)
                total_tokens += prompt_length
                
                # Stop if we've reached max_tokens
                if total_tokens >= args.max_tokens:
                    break
            
            logger.info(
                f"Selected {len(selected_prompts)} prompts with ~{total_tokens} tokens "
                f"for {concept_category_name}"
            )
            
            # Tokenize all selected prompts together
            input_ids = tokenizer(
                selected_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=args.max_tokens,
            ).to(device)
            input_ids = input_ids.input_ids

            vector_dim = concept_vectors.shape[1]

            # Generate and save a random vector for this concept (reuse if exists)
            random_vector_dir = f"assets/velocity_direction/{model_name}/random_vectors"
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

            # Run velocity direction analysis with concept vector and random vector
            for vector_type, vector_source in [
                ("concept", concept_vectors),
                ("random", random_vector),
            ]:
                results: dict[int, dict[str, torch.Tensor]] = {}

                for layer_idx in tqdm(
                    layers_to_run,
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
                        eps_rel=args.eps_rel,
                        eps_min=args.eps_min,
                        eps_max=args.eps_max,
                    )

                    results[layer_idx] = layer_results

                # Save results
                save_path = f"assets/velocity_direction/{model_name}/velocity_direction_{concept_category_name}_{vector_type}.pt"
                torch.save(
                    {
                        "model": model_full_name,
                        "concept_category": concept_category_name,
                        "vector_type": vector_type,
                        "alpha_min": args.alpha_min,
                        "alpha_max": args.alpha_max,
                        "alpha_points": args.alpha_points,
                        "eps_rel": args.eps_rel,
                        "eps_min": args.eps_min,
                        "eps_max": args.eps_max,
                        "results": results,
                    },
                    save_path,
                )
                logger.info(f"Saved velocity direction ({vector_type}) to {save_path}")
        
        # Clean up model to free memory before next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logger.info(f"Finished processing model: {model_full_name}")


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
    eps_rel: float = 1e-2,
    eps_min: float = 1e-5,
    eps_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Compute velocity direction consistency using backward-epsilon velocities.
    
    For each alpha, we compute:
    - v_forward(α) = h(α) - h(α-ε): forward velocity
    - v_backward(α) = h(α-ε) - h(α-2ε): backward velocity
    - cos(v_forward, v_backward): measures direction change
    - cos(h(α), h(α-ε)): similarity between consecutive hidden states
    - cos(h(α-ε), h(α-2ε)): similarity between consecutive hidden states
    
    This measures whether the trajectory is curving in hidden space:
    - cos(velocity) ≈ 1: velocity direction is stable (straight trajectory)
    - cos(velocity) < 1: velocity direction is changing (curved trajectory)
    - cos(hidden states) ≈ 1: hidden states are similar (slow change)
    - cos(hidden states) < 1: hidden states are diverging (fast change)
    
    The epsilon is chosen adaptively as: eps = clamp(alpha * eps_rel, eps_min, eps_max)
    
    Returns:
        Dictionary containing:
        - alpha: tensor of alpha values (where alpha - 2*eps > 0)
        - eps: tensor of epsilon values used for each alpha
        - cos_velocity: mean cos(v_forward, v_backward) 
        - cos_velocity_std: std cos(v_forward, v_backward) across tokens
        - cos_x0_x1: mean cos(h(α), h(α-ε))
        - cos_x0_x1_std: std cos(h(α), h(α-ε)) across tokens
        - cos_x1_x2: mean cos(h(α-ε), h(α-2ε))
        - cos_x1_x2_std: std cos(h(α-ε), h(α-2ε)) across tokens
        - layer_idx: the layer index
    """
    # Build alpha sweep in log-space
    alphas = torch.logspace(
        float(torch.log10(torch.tensor(alpha_min))),
        float(torch.log10(torch.tensor(alpha_max))),
        steps=int(alpha_points),
    ).tolist()

    def _run_with_alpha(alpha_value: float) -> torch.Tensor:
        """Run model with steering and capture last layer hidden states."""
        return run_model_with_steering(
            model=model,
            input_ids=input_ids,
            steering_vector=steering_vector,
            layer_idx=layer_idx,
            alpha_value=alpha_value,
            device=device,
        )

    def _cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        """Compute cosine similarity between batch of vectors.
        
        Args:
            a: tensor of shape [N, d]
            b: tensor of shape [N, d]
        
        Returns:
            Tuple of (mean cosine similarity, std cosine similarity)
        """
        a_norm = torch.nn.functional.normalize(a, dim=-1)
        b_norm = torch.nn.functional.normalize(b, dim=-1)
        cos_sim = (a_norm * b_norm).sum(dim=-1)  # [N]
        return cos_sim.mean().item(), cos_sim.std().item()

    # Compute velocity direction consistency for each alpha
    alpha_out = []
    eps_out = []
    cos_velocity_list = []
    cos_velocity_std_list = []
    cos_x0_x1_list = []
    cos_x0_x1_std_list = []
    cos_x1_x2_list = []
    cos_x1_x2_std_list = []

    for alpha in tqdm(alphas, desc=f"  Layer {layer_idx}", leave=False):
        # Compute adaptive epsilon
        eps = float(max(eps_min, min(eps_max, abs(alpha) * eps_rel)))
        
        # Require alpha - 2*eps > 0 to compute both velocities
        if alpha - 2.0 * eps <= 0.0:
            continue
        
        # Compute hidden states at three points: alpha, alpha-eps, alpha-2eps
        # x0 = h(alpha)
        x0 = hidden_to_flat(_run_with_alpha(alpha))  # [batch*seq, d]
        # x1 = h(alpha - eps)
        x1 = hidden_to_flat(_run_with_alpha(alpha - eps))  # [batch*seq, d]
        # x2 = h(alpha - 2*eps)
        x2 = hidden_to_flat(_run_with_alpha(alpha - 2.0 * eps))  # [batch*seq, d]
        
        # Compute velocities
        # v_forward(α) = h(α) - h(α-ε)
        v_forward = x0 - x1  # [batch*seq, d]
        # v_backward(α) = h(α-ε) - h(α-2ε)
        v_backward = x1 - x2  # [batch*seq, d]
        
        # Compute cosine similarity between the two velocity vectors
        cos_v_mean, cos_v_std = _cosine_similarity_batch(v_forward, v_backward)
        
        # Compute cosine similarity between hidden states
        # cos(x0, x1): similarity between h(α) and h(α-ε)
        cos_x0_x1_mean, cos_x0_x1_std = _cosine_similarity_batch(x0, x1)
        # cos(x1, x2): similarity between h(α-ε) and h(α-2ε)
        cos_x1_x2_mean, cos_x1_x2_std = _cosine_similarity_batch(x1, x2)
        
        alpha_out.append(alpha)
        eps_out.append(eps)
        cos_velocity_list.append(cos_v_mean)
        cos_velocity_std_list.append(cos_v_std)
        cos_x0_x1_list.append(cos_x0_x1_mean)
        cos_x0_x1_std_list.append(cos_x0_x1_std)
        cos_x1_x2_list.append(cos_x1_x2_mean)
        cos_x1_x2_std_list.append(cos_x1_x2_std)

    return {
        "alpha": torch.tensor(alpha_out, dtype=torch.float32),
        "eps": torch.tensor(eps_out, dtype=torch.float32),
        "cos_velocity": torch.tensor(cos_velocity_list, dtype=torch.float32),
        "cos_velocity_std": torch.tensor(cos_velocity_std_list, dtype=torch.float32),
        "cos_x0_x1": torch.tensor(cos_x0_x1_list, dtype=torch.float32),
        "cos_x0_x1_std": torch.tensor(cos_x0_x1_std_list, dtype=torch.float32),
        "cos_x1_x2": torch.tensor(cos_x1_x2_list, dtype=torch.float32),
        "cos_x1_x2_std": torch.tensor(cos_x1_x2_std_list, dtype=torch.float32),
        "layer_idx": layer_idx,
    }


if __name__ == "__main__":
    velocity_direction()
