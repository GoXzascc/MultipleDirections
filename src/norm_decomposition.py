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


def norm_decomposition():
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
    parser.add_argument(
        "--layers",
        type=str,
        default="25,50,75",
        help="Comma-separated percentages (e.g., '25,50,75') or layer indices (e.g., '5,10,15'). Use 'all' to run all layers.",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/norm_decomposition.log")
    logger.info(f"args: {args}")
    set_seed(args.seed)
    
    # Determine which models to process
    models_to_process = (
        [(args.model, MODEL_LAYERS[args.model])]
        if args.model is not None
        else list(MODEL_LAYERS.items())
    )
    
    logger.info(f"Starting norm decomposition measurement...")
    logger.info(f"Models to process: {[model_name for model_name, _ in models_to_process]}")
    device = "cuda"
    dtype = torch.float32
    
    for model_full_name, max_layers in models_to_process:
        logger.info(f"Processing model: {model_full_name}")
        model_name = get_model_name_for_path(model_full_name)
        os.makedirs(f"assets/norm_decomposition/{model_name}", exist_ok=True)
        
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
            random_vector_dir = f"assets/norm_decomposition/{model_name}/random_vectors"
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
                    layers_to_run,
                    desc=f"Measuring norm decomposition ({concept_category_name}, {vector_type})",
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
                save_path = f"assets/norm_decomposition/{model_name}/norm_decomposition_{concept_category_name}_{vector_type}.pt"
                torch.save(
                    {
                        "model": model_full_name,
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
        
        # Clean up model to free memory before next model
        del model
        del tokenizer
        torch.cuda.empty_cache()
        logger.info(f"Finished processing model: {model_full_name}")


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
        - total_norm_A_std: std per-token ||h(α) - αv||
        - parallel_norm_A: average per-token ||projection onto v||
        - parallel_norm_A_std: std per-token ||projection onto v||
        - ortho_norm_A: average per-token ||orthogonal component||
        - ortho_norm_A_std: std per-token ||orthogonal component||
        
        Residual B (with intercept): residual_B = (h(α) - h(0)) - αv
        - total_norm_B: average per-token ||(h(α) - h(0)) - αv||
        - total_norm_B_std: std per-token ||(h(α) - h(0)) - αv||
        - parallel_norm_B: average per-token ||projection onto v||
        - parallel_norm_B_std: std per-token ||projection onto v||
        - ortho_norm_B: average per-token ||orthogonal component||
        - ortho_norm_B_std: std per-token ||orthogonal component||
        
        - h0_norm: average per-token ||h(0)|| (baseline hidden state norm)
        - h0_norm_std: std per-token ||h(0)|| (baseline hidden state norm)
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

    # Normalize steering vector for projection computations (use float32)
    steering_vec_f32 = steering_vector.to(dtype=torch.float32, device=device)
    steering_vec_normalized = steering_vec_f32 / steering_vec_f32.norm()

    # Get baseline hidden state (alpha = 0)
    h0 = hidden_to_flat(_run_with_alpha(0.0))  # [batch*seq, d]
    h0 = h0.to(dtype=torch.float32)  # Ensure float32
    h0_f64 = h0.to(dtype=torch.float64)  # Pre-compute float64 version for subtraction
    h0_norms = torch.norm(h0, p=2, dim=1)  # [batch*seq]
    h0_norm_mean = h0_norms.mean().item()
    h0_norm_std = h0_norms.std().item()

    alpha_list = []
    # h(α) norm statistics
    h_alpha_norm_list = []
    h_alpha_norm_std_list = []
    # residual_A = h(α) - αv (no intercept, checking if h passes through origin)
    total_norm_A_list = []
    total_norm_A_std_list = []
    parallel_norm_A_list = []
    parallel_norm_A_std_list = []
    ortho_norm_A_list = []
    ortho_norm_A_std_list = []
    # residual_B = (h(α) - h(0)) - αv (with intercept, checking if h(α) ≈ h(0) + αv)
    total_norm_B_list = []
    total_norm_B_std_list = []
    parallel_norm_B_list = []
    parallel_norm_B_std_list = []
    ortho_norm_B_list = []
    ortho_norm_B_std_list = []

    for alpha in tqdm(alphas, desc=f"  Layer {layer_idx}", leave=False):
        # Get hidden state with steering
        h_alpha = hidden_to_flat(_run_with_alpha(alpha))  # [batch*seq, d]
        h_alpha = h_alpha.to(dtype=torch.float32)  # Ensure float32
        
        # Compute h(α) norm statistics
        h_alpha_norms = torch.norm(h_alpha, p=2, dim=1)  # [batch*seq]
        h_alpha_norm_mean = h_alpha_norms.mean().item()
        h_alpha_norm_std = h_alpha_norms.std().item()
        
        # Compute steering contribution (float32)
        steering_contribution = alpha * steering_vec_f32.unsqueeze(0)  # [1, d]
        
        # ===== Residual A: h(α) - αv (no intercept) =====
        # Use double precision for subtraction to minimize numerical errors
        h_alpha_f64 = h_alpha.to(dtype=torch.float64)
        steering_contribution_f64 = steering_contribution.to(dtype=torch.float64)
        residual_A = (h_alpha_f64 - steering_contribution_f64).to(dtype=torch.float32)  # [batch*seq, d]
        
        # Project residual_A onto steering direction
        dot_A = torch.matmul(residual_A, steering_vec_normalized)  # [batch*seq]
        parallel_A = dot_A.unsqueeze(1) * steering_vec_normalized.unsqueeze(0)  # [batch*seq, d]
        ortho_A = residual_A - parallel_A  # [batch*seq, d]
        
        # Compute mean and std norms for residual_A
        total_norm_A_per_token = torch.norm(residual_A, p=2, dim=1)  # [batch*seq]
        parallel_norm_A_per_token = torch.norm(parallel_A, p=2, dim=1)  # [batch*seq]
        ortho_norm_A_per_token = torch.norm(ortho_A, p=2, dim=1)  # [batch*seq]
        
        total_norm_A = total_norm_A_per_token.mean().item()
        total_norm_A_std = total_norm_A_per_token.std().item()
        parallel_norm_A = parallel_norm_A_per_token.mean().item()
        parallel_norm_A_std = parallel_norm_A_per_token.std().item()
        ortho_norm_A = ortho_norm_A_per_token.mean().item()
        ortho_norm_A_std = ortho_norm_A_per_token.std().item()
        
        # ===== Residual B: (h(α) - h(0)) - αv (with intercept) =====
        # Use double precision for subtraction to minimize numerical errors
        hs_diff = (h_alpha_f64 - h0_f64).to(dtype=torch.float32)  # [batch*seq, d]
        hs_diff_f64 = hs_diff.to(dtype=torch.float64)
        residual_B = (hs_diff_f64 - steering_contribution_f64).to(dtype=torch.float32)  # [batch*seq, d]
        
        
        # Project residual_B onto steering direction
        dot_B = torch.matmul(residual_B, steering_vec_normalized)  # [batch*seq]
        parallel_B = dot_B.unsqueeze(1) * steering_vec_normalized.unsqueeze(0)  # [batch*seq, d]
        ortho_B = residual_B - parallel_B  # [batch*seq, d]
        
        # Compute mean and std norms for residual_B
        total_norm_B_per_token = torch.norm(residual_B, p=2, dim=1)  # [batch*seq]
        parallel_norm_B_per_token = torch.norm(parallel_B, p=2, dim=1)  # [batch*seq]
        ortho_norm_B_per_token = torch.norm(ortho_B, p=2, dim=1)  # [batch*seq]
        
        total_norm_B = total_norm_B_per_token.mean().item()
        total_norm_B_std = total_norm_B_per_token.std().item()
        parallel_norm_B = parallel_norm_B_per_token.mean().item()
        parallel_norm_B_std = parallel_norm_B_per_token.std().item()
        ortho_norm_B = ortho_norm_B_per_token.mean().item()
        ortho_norm_B_std = ortho_norm_B_per_token.std().item()
        
        alpha_list.append(alpha)
        h_alpha_norm_list.append(h_alpha_norm_mean)
        h_alpha_norm_std_list.append(h_alpha_norm_std)
        total_norm_A_list.append(total_norm_A)
        total_norm_A_std_list.append(total_norm_A_std)
        parallel_norm_A_list.append(parallel_norm_A)
        parallel_norm_A_std_list.append(parallel_norm_A_std)
        ortho_norm_A_list.append(ortho_norm_A)
        ortho_norm_A_std_list.append(ortho_norm_A_std)
        total_norm_B_list.append(total_norm_B)
        total_norm_B_std_list.append(total_norm_B_std)
        parallel_norm_B_list.append(parallel_norm_B)
        parallel_norm_B_std_list.append(parallel_norm_B_std)
        ortho_norm_B_list.append(ortho_norm_B)
        ortho_norm_B_std_list.append(ortho_norm_B_std)

    return {
        "alpha": torch.tensor(alpha_list, dtype=torch.float32),
        # h(α) norm statistics
        "h_alpha_norm": torch.tensor(h_alpha_norm_list, dtype=torch.float32),
        "h_alpha_norm_std": torch.tensor(h_alpha_norm_std_list, dtype=torch.float32),
        # Residual A: h(α) - αv (no intercept)
        "total_norm_A": torch.tensor(total_norm_A_list, dtype=torch.float32),
        "total_norm_A_std": torch.tensor(total_norm_A_std_list, dtype=torch.float32),
        "parallel_norm_A": torch.tensor(parallel_norm_A_list, dtype=torch.float32),
        "parallel_norm_A_std": torch.tensor(parallel_norm_A_std_list, dtype=torch.float32),
        "ortho_norm_A": torch.tensor(ortho_norm_A_list, dtype=torch.float32),
        "ortho_norm_A_std": torch.tensor(ortho_norm_A_std_list, dtype=torch.float32),
        # Residual B: (h(α) - h(0)) - αv (with intercept)
        "total_norm_B": torch.tensor(total_norm_B_list, dtype=torch.float32),
        "total_norm_B_std": torch.tensor(total_norm_B_std_list, dtype=torch.float32),
        "parallel_norm_B": torch.tensor(parallel_norm_B_list, dtype=torch.float32),
        "parallel_norm_B_std": torch.tensor(parallel_norm_B_std_list, dtype=torch.float32),
        "ortho_norm_B": torch.tensor(ortho_norm_B_list, dtype=torch.float32),
        "ortho_norm_B_std": torch.tensor(ortho_norm_B_std_list, dtype=torch.float32),
        # Baseline hidden state norm
        "h0_norm": torch.tensor(h0_norm_mean, dtype=torch.float32),
        "h0_norm_std": torch.tensor(h0_norm_std, dtype=torch.float32),
        "layer_idx": layer_idx,
    }


if __name__ == "__main__":
    norm_decomposition()
