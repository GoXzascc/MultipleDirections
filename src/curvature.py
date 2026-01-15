import argparse
import torch
import transformers
import os
import random
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


def curvature():
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
    
    parser.add_argument("--eps_max", type=float, default=1)
    parser.add_argument("--denom_eps", type=float, default=1e-5)
    parser.add_argument(
        "--layers",
        type=str,
        default="25,50,75",
        help="Comma-separated percentages (e.g., '25,50,75') or layer indices (e.g., '5,10,15'). Use 'all' to run all layers.",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/curvature.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    
    # Determine which models to process
    models_to_process = (
        [(args.model, MODEL_LAYERS[args.model])]
        if args.model is not None
        else list(MODEL_LAYERS.items())
    )
    
    logger.info(f"Starting curvature measurement...")
    logger.info(f"Models to process: {[model_name for model_name, _ in models_to_process]}")
    device = "cuda"
    dtype = torch.float32
    
    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        model_name = get_model_name_for_path(model_full_name)
        os.makedirs(f"assets/curvature/{model_name}", exist_ok=True)
        
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

            num_layers = min(max_layers, concept_vectors.shape[0])
            vector_dim = concept_vectors.shape[1]

            # Generate and save a random vector for this concept (reuse if exists)
            random_vector_dir = f"assets/curvature/{model_name}/random_vectors"
            os.makedirs(random_vector_dir, exist_ok=True)
            random_vector_path = f"{random_vector_dir}/{concept_category_name}.pt"

            if os.path.exists(random_vector_path):
                random_vector_data = torch.load(random_vector_path)
                random_vector = random_vector_data["random_vector"]
                logger.info(f"Using saved random vector for {concept_category_name}")
            else:
                # Generate a random vector with the same dimension as concept vectors
                random_vector = torch.randn(vector_dim, dtype=torch.float32)
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

            # Run steering with concept vector and random vector separately
            # For each vector type, run with remove_steering_direction=True and False
            for vector_type, vector_source in [
                ("concept", concept_vectors),
                ("random", random_vector),
            ]:
                for remove_flag in [False, True]:
                    remove_suffix = "w_remove" if remove_flag else "wo_remove"
                    results: dict[int, dict[str, torch.Tensor]] = {}

                    for layer_idx in tqdm(
                        layers_to_run,
                        desc=f"Measuring curvature ({concept_category_name}, {vector_type}, {remove_suffix})",
                    ):
                        if vector_type == "concept":
                            # Use the concept vector from the corresponding layer
                            steering_vector = concept_vectors[layer_idx, :]
                        else:
                            # Use the same random vector for all layers
                            steering_vector = random_vector

                        alpha_list: list[torch.Tensor] = []
                        eps_list: list[torch.Tensor] = []
                        kappa_list: list[torch.Tensor] = []
                        kappa_std_list: list[torch.Tensor] = []

                        for item in model_steering(
                            model=model,
                            input_ids=input_ids,
                            concept_vector=steering_vector,
                            layer_idx=layer_idx,
                            dtype=dtype,
                            device=device,
                            alpha_factor_init=0,
                            alpha_factor_num=0,
                            alpha_min=args.alpha_min,
                            alpha_max=args.alpha_max,
                            alpha_points=args.alpha_points,
                            eps_rel=args.eps_rel,
                            eps_min=args.eps_min,
                            eps_max=args.eps_max,
                            denom_eps=args.denom_eps,
                            remove_steering_direction=remove_flag,
                        ):
                            alpha_list.append(item["alpha"])
                            eps_list.append(item["eps"])
                            kappa_list.append(item["kappa"])
                            kappa_std_list.append(item["kappa_std"])

                        results[layer_idx] = {
                            "alpha": (
                                torch.stack(alpha_list) if alpha_list else torch.empty((0,))
                            ),
                            "eps": torch.stack(eps_list) if eps_list else torch.empty((0,)),
                            "kappa": (
                                torch.stack(kappa_list) if kappa_list else torch.empty((0,))
                            ),
                            "kappa_std": (
                                torch.stack(kappa_std_list) if kappa_std_list else torch.empty((0,))
                            ),
                            "layer_idx": layer_idx,
                        }

                    # Save results with different names for concept/random and wo_remove/w_remove
                    save_path = f"assets/curvature/{model_name}/curvature_{concept_category_name}_{vector_type}_{remove_suffix}.pt"
                    torch.save(
                        {
                            "model": model_full_name,
                            "concept_category": concept_category_name,
                            "vector_type": vector_type,
                            "remove_steering_direction": remove_flag,
                            "alpha_min": args.alpha_min,
                            "alpha_max": args.alpha_max,
                            "alpha_points": args.alpha_points,
                            "eps_rel": args.eps_rel,
                            "eps_min": args.eps_min,
                            "eps_max": args.eps_max,
                            "denom_eps": args.denom_eps,
                            "results": results,
                        },
                        save_path,
                    )
                    logger.info(
                        f"Saved curvature sweep ({vector_type}, {remove_suffix}) to {save_path}"
                    )
        
        # Clean up model to free memory before next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logger.info(f"Finished processing model: {model_full_name}")


def model_steering(
    model,
    input_ids,
    concept_vector,
    layer_idx,
    dtype,
    device,
    alpha_factor_init,  # unused, kept for API compatibility
    alpha_factor_num,  # unused, kept for API compatibility
    alpha_min: float = 1e-3,
    alpha_max: float = 1e7,
    alpha_points: int = 200,
    eps_rel: float = 1e-2,
    eps_min: float = 1e-9,
    eps_max: float = 1.0,
    denom_eps: float = 1e-5,
    remove_steering_direction: bool = False,
):
    """
    Sweep alphas and compute discrete curvature with a backward epsilon:

      v(alpha) = (x(alpha) - x(alpha - eps)) / eps
      kappa(alpha) = ||v(alpha) - v(alpha - eps)|| / (||v(alpha)|| + ||v(alpha - eps)|| + denom_eps)

    Here eps is chosen per alpha as:
      eps = clamp(alpha * eps_rel, eps_min, eps_max)

    Notes:
    - x(alpha) is the reshaped last-block hidden state [batch, seq, d] -> [batch*seq, d].
      No averaging is performed to preserve all token-level information.
    - Curvature is computed per token, then aggregated (mean) across all tokens.
    - We require alpha - 2*eps > 0 to form v(alpha - eps).
    - Computation uses float64 (double) for numerical precision.
    """

    # Build alpha sweep in linear alpha-space, but spaced in log-space for coverage
    alphas = torch.logspace(
        float(torch.log10(torch.tensor(alpha_min))),
        float(torch.log10(torch.tensor(alpha_max))),
        steps=int(alpha_points),
    ).tolist()

    for alpha in alphas:
        eps = float(max(eps_min, min(eps_max, abs(alpha) * eps_rel)))
        if alpha - 2.0 * eps <= 0.0:
            continue

        # Backward-eps velocities:
        # v(alpha) depends on x(alpha) and x(alpha-eps)
        # v(alpha-eps) depends on x(alpha-eps) and x(alpha-2eps)
        # x0, x1, x2 are [batch*seq, d] - all token information preserved
        # All computations use float64 for numerical precision
        x0 = hidden_to_flat(
            run_model_with_steering(
                model=model,
                input_ids=input_ids,
                steering_vector=concept_vector,
                layer_idx=layer_idx,
                alpha_value=alpha,
                device=device,
            ),
            target_dtype=torch.float64,
        )
        x1 = hidden_to_flat(
            run_model_with_steering(
                model=model,
                input_ids=input_ids,
                steering_vector=concept_vector,
                layer_idx=layer_idx,
                alpha_value=alpha - eps,
                device=device,
            ),
            target_dtype=torch.float64,
        )
        x2 = hidden_to_flat(
            run_model_with_steering(
                model=model,
                input_ids=input_ids,
                steering_vector=concept_vector,
                layer_idx=layer_idx,
                alpha_value=alpha - 2.0 * eps,
                device=device,
            ),
            target_dtype=torch.float64,
        )

        # Convert eps to float64 tensor for consistent precision
        eps_tensor = torch.tensor(eps, dtype=torch.float64, device=x0.device)

        # Compute velocities: [batch*seq, d], float64
        if remove_steering_direction:
            # Remove the steering direction component from the velocity
            # concept_vector is [d], need to expand to [batch*seq, d] for broadcasting
            concept_vec_expanded = (
                concept_vector.to(dtype=torch.float64, device=x0.device)
                .unsqueeze(0)
                .expand_as(x0)
            )
            v_alpha = (
                x0 - x1 - eps_tensor * concept_vec_expanded
            ) / eps_tensor  # [batch*seq, d]
            v_prev = (
                x1 - x2 - eps_tensor * concept_vec_expanded
            ) / eps_tensor  # [batch*seq, d] - v(alpha - eps)
        else:
            v_alpha = (x0 - x1) / eps_tensor  # [batch*seq, d]
            v_prev = (x1 - x2) / eps_tensor  # [batch*seq, d] - v(alpha - eps)

        # Compute curvature: L2 norm over last dim, then average over tokens
        # All computations in float64
        num = torch.norm(v_alpha - v_prev, p=2, dim=-1)  # [batch*seq]
        den = (
            torch.norm(v_alpha, p=2, dim=-1)
            + torch.norm(v_prev, p=2, dim=-1)
            + torch.tensor(denom_eps, dtype=torch.float64, device=x0.device)
        )  # [batch*seq]
        kappa_per_token = num / den  # [batch*seq] - curvature per token
        
        # Compute mean and std across tokens
        kappa_mean = kappa_per_token.mean().item()  # scalar, float64 -> Python float
        kappa_std = kappa_per_token.std().item()  # scalar, float64 -> Python float

        yield {
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "eps": torch.tensor(eps, dtype=torch.float32),
            "kappa": torch.tensor(kappa_mean, dtype=torch.float32),
            "kappa_std": torch.tensor(kappa_std, dtype=torch.float32),
        }


if __name__ == "__main__":
    curvature()
